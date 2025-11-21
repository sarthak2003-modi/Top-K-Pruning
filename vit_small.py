"""
ViT-Small on ImageNet-1K with 2:4 Structured Sparsity (Top-2 in every 4)

This script applies 2:4 structured pruning to ViT-small:
- Groups every 4 contiguous weights 
- Keeps only top-2 weights per group (k=2 out of 4)
- Two-stage training: Stage 1 with auxiliary loss, Stage 2 with hard masks

Uses pretrained ViT-small from timm library.
"""

import argparse
import os
import random
from time import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

try:
    import timm
    from timm.data import create_transform
    from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False
    print("Warning: timm not installed. Install with: pip install timm")


def parse_args():
    parser = argparse.ArgumentParser(
        description='ViT-Small with 2:4 structured sparsity on ImageNet-1K'
    )
    
    # Model and data
    parser.add_argument('--model', type=str, default='vit_small_patch16_224.augreg_in21k_ft_in1k',
                        help='Model name (default: vit_small_patch16_224.augreg_in21k_ft_in1k for 78.55% baseline)')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained weights')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to ImageNet dataset')
    parser.add_argument('--num-classes', type=int, default=1000,
                        help='Number of classes (default: 1000)')
    
    # 2:4 Sparsity configuration
    parser.add_argument('--sparsity-type', type=str, default='2:4',
                        help='Sparsity pattern (default: 2:4)')
    parser.add_argument('--reg-type', type=str, default='exponential',
                        choices=['none', 'l2', 'exponential'],
                        help='Regularizer type (default: exponential)')
    parser.add_argument('--beta', type=float, default=1000,
                        help='Weight on sparsity auxiliary loss (default: 1000)')
    parser.add_argument('--alpha', type=float, default=2.0,
                        help='Exponential falloff parameter (default: 2.0)')
    parser.add_argument('--target-layers', type=str, default='qkv,proj,mlp',
                        help='Which layers to prune: qkv,proj,mlp (default: all)')
    
    # Training configuration
    parser.add_argument('--epochs-stage1', type=int, default=30,
                        help='Number of epochs for stage 1 (default: 30)')
    parser.add_argument('--epochs-stage2', type=int, default=30,
                        help='Number of epochs for stage 2 (default: 30)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='Weight decay (default: 0.05)')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size (default: 256)')
    parser.add_argument('--grad-accum', type=int, default=1,
                        help='Gradient accumulation steps (default: 1)')
    
    # Optimization settings
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable mixed precision training')
    parser.add_argument('--val-freq', type=int, default=5,
                        help='Validate every N epochs (default: 5)')
    parser.add_argument('--aux-loss-freq', type=int, default=1,
                        help='Compute aux loss every N steps (default: 1)')
    
    # Other settings
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Number of data loading workers (default: 8)')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints_vit',
                        help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    
    return parser.parse_args()


# ================== 2:4 Structured Sparsity Functions ==================

def get_linear_layers_to_prune(model, target_layers):
    """
    Get all Linear layers in ViT that should be pruned.
    
    ViT-small architecture:
    - 12 transformer blocks, each with:
        - qkv projection (combines Q, K, V)
        - attention output projection
        - MLP fc1 and fc2
    """
    target_set = set(target_layers.split(','))
    layers_to_prune = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if this layer matches our target criteria
            should_prune = False
            
            if 'qkv' in target_set and 'qkv' in name:
                should_prune = True
            elif 'proj' in target_set and ('proj' in name or 'attn.proj' in name):
                should_prune = True
            elif 'mlp' in target_set and ('mlp.fc' in name or 'mlp.linear' in name):
                should_prune = True
            
            if should_prune:
                layers_to_prune.append((name, module))
    
    return layers_to_prune


def apply_2_4_mask(weight: torch.Tensor) -> torch.Tensor:
    """
    Apply 2:4 structured sparsity mask.
    Groups every 4 contiguous elements and keeps top-2.
    
    Args:
        weight: Tensor of shape [out_features, in_features]
    
    Returns:
        mask: Binary mask of same shape
    """
    out_features, in_features = weight.shape
    
    # Check if divisible by 4
    if in_features % 4 != 0:
        # Pad to nearest multiple of 4
        pad_size = (4 - in_features % 4) % 4
        weight_padded = torch.nn.functional.pad(weight, (0, pad_size), value=0)
    else:
        weight_padded = weight
        pad_size = 0
    
    # Reshape to [out_features, in_features//4, 4]
    weight_grouped = weight_padded.view(out_features, -1, 4)
    
    # Get top-2 indices in each group of 4
    _, topk_idx = torch.topk(weight_grouped.abs(), k=2, dim=2, sorted=False)
    
    # Create mask
    mask_grouped = torch.zeros_like(weight_grouped)
    mask_grouped.scatter_(2, topk_idx, 1.0)
    
    # Reshape back and remove padding
    mask = mask_grouped.view(out_features, -1)
    if pad_size > 0:
        mask = mask[:, :-pad_size]
    
    return mask


@torch.no_grad()
def make_2_4_mask(weight: torch.Tensor) -> torch.Tensor:
    """Wrapper for 2:4 mask creation."""
    return apply_2_4_mask(weight)


def sparsity_aux_loss_2_4_l2(model: nn.Module, layers_to_prune, alpha: float = 0.3):
    """
    L2 regularization on weights beyond top-2 in each group of 4.
    Normalized by total number of groups.
    """
    total_penalty = 0.0
    total_groups = 0
    
    for name, module in layers_to_prune:
        if isinstance(module, nn.Linear):
            W = module.weight  # [out_features, in_features]
            out_features, in_features = W.shape
            
            # Pad if necessary
            if in_features % 4 != 0:
                pad_size = (4 - in_features % 4) % 4
                W_padded = torch.nn.functional.pad(W, (0, pad_size), value=0)
            else:
                W_padded = W
                pad_size = 0
            
            # Reshape to groups of 4
            W_grouped = W_padded.view(out_features, -1, 4)
            W_grouped_sq = W_grouped * W_grouped
            
            # Get top-2 indices per group
            _, topk_idx = torch.topk(W_grouped.abs(), k=2, dim=2, sorted=False)
            
            # Create mask: 1 for non-top-2, 0 for top-2
            mask = torch.ones_like(W_grouped_sq)
            mask.scatter_(2, topk_idx, 0.0)
            
            # L2 penalty on non-top-2 weights
            layer_penalty = (W_grouped_sq * mask).sum()
            total_penalty += layer_penalty
            total_groups += out_features * W_grouped.size(1)
    
    if total_groups == 0:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    
    return total_penalty / total_groups


def sparsity_aux_loss_2_4_exponential(model: nn.Module, layers_to_prune, alpha: float = 2.0):
    """
    Exponential falloff regularization for 2:4 sparsity.
    Uses rank-based exponential decay within each group of 4.
    """
    total_penalty = 0.0
    total_groups = 0
    
    for name, module in layers_to_prune:
        if isinstance(module, nn.Linear):
            W = module.weight  # [out_features, in_features]
            out_features, in_features = W.shape
            
            # Pad if necessary
            if in_features % 4 != 0:
                pad_size = (4 - in_features % 4) % 4
                W_padded = torch.nn.functional.pad(W, (0, pad_size), value=0)
            else:
                W_padded = W
                pad_size = 0
            
            # Reshape to groups of 4
            W_grouped = W_padded.view(out_features, -1, 4)
            W_grouped_sq = W_grouped * W_grouped
            
            # Sort squared weights in descending order per group
            vals_sq, _ = torch.sort(W_grouped_sq, dim=2, descending=True)
            
            # Create rank-based exponential weights
            # Ranks: [0, 1, 2, 3], we want to penalize ranks 2 and 3 (beyond top-2)
            ranks = torch.arange(4, device=W.device)
            exp_weights = torch.where(
                ranks >= 2,  # Beyond top-2
                torch.exp(-alpha * (ranks - 2).float()),
                torch.zeros_like(ranks, dtype=W.dtype)
            )
            
            # Apply exponential penalty
            layer_penalty = (vals_sq * exp_weights.unsqueeze(0).unsqueeze(0)).sum()
            total_penalty += layer_penalty
            total_groups += out_features * W_grouped.size(1)
    
    if total_groups == 0:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    
    return total_penalty / total_groups


def get_aux_loss_fn(reg_type: str):
    """Returns the appropriate auxiliary loss function."""
    if reg_type == 'none':
        return lambda model, layers, alpha: torch.tensor(0.0, device=next(model.parameters()).device)
    elif reg_type == 'l2':
        return sparsity_aux_loss_2_4_l2
    elif reg_type == 'exponential':
        return sparsity_aux_loss_2_4_exponential
    else:
        raise ValueError(f"Unknown reg_type: {reg_type}")


def apply_2_4_mask_to_model(model, layers_to_prune):
    """
    Apply 2:4 structured sparsity masks to specified layers.
    """
    total_kept, total_elems = 0, 0
    
    for name, module in layers_to_prune:
        if isinstance(module, nn.Linear):
            W = module.weight.data
            mask = make_2_4_mask(W).to(W.device)
            
            # Apply hard mask
            module.weight.data.mul_(mask)
            
            kept_here = mask.sum().item()
            total_kept += kept_here
            total_elems += W.numel()
            
            # Store mask as buffer
            if hasattr(module, "weight_mask"):
                module.weight_mask.copy_(mask)
            else:
                module.register_buffer("weight_mask", mask)
            
            # Register gradient hook
            def _make_mask_grad(mask_ref):
                def _hook(grad):
                    return grad * mask_ref
                return _hook
            
            module.weight.register_hook(_make_mask_grad(module.weight_mask))
    
    density = 100.0 * total_kept / max(1, total_elems)
    print(f"[Mask] Applied 2:4 sparsity: kept {total_kept}/{total_elems} "
          f"({density:.2f}% density, target ~50%).")


def sparsity_report(model, layers_to_prune):
    """Print sparsity statistics for pruned layers."""
    print("\n" + "="*80)
    print("SPARSITY REPORT (2:4 Structured)")
    print("="*80)
    
    total_params = 0
    total_zeros = 0
    
    for name, module in layers_to_prune:
        if isinstance(module, nn.Linear):
            w = module.weight.data
            n_params = w.numel()
            n_zeros = (w == 0).sum().item()
            sparsity = 100.0 * n_zeros / n_params
            
            total_params += n_params
            total_zeros += n_zeros
            
            print(f"{name:50s} | Params: {n_params:10d} | Zeros: {n_zeros:10d} | "
                  f"Sparsity: {sparsity:6.2f}%")
    
    overall_sparsity = 100.0 * total_zeros / max(1, total_params)
    print("="*80)
    print(f"{'OVERALL (Pruned layers)':50s} | Params: {total_params:10d} | "
          f"Zeros: {total_zeros:10d} | Sparsity: {overall_sparsity:6.2f}%")
    print("="*80 + "\n")


# ================== Training Functions ==================

def train_one_epoch(model, loader, optimizer, scheduler, scaler, epoch, epochs, device,
                   aux_loss_fn, layers_to_prune, beta, alpha, use_amp, grad_accum, 
                   aux_loss_freq):
    """Train for one epoch with gradient accumulation and mixed precision."""
    model.train()
    run = {"loss": 0.0, "ce": 0.0, "aux": 0.0, "acc": 0.0, "n": 0}
    ce_loss = nn.CrossEntropyLoss()
    
    optimizer.zero_grad()
    
    for step, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        
        with autocast(enabled=use_amp):
            logits = model(x)
            ce = ce_loss(logits, y)
            
            # Compute auxiliary loss less frequently
            if step % aux_loss_freq == 0 and beta > 0:
                aux = aux_loss_fn(model, layers_to_prune, alpha=alpha)
            else:
                aux = torch.tensor(0.0, device=device)
            
            loss = ce + beta * aux
            loss = loss / grad_accum
        
        scaler.scale(loss).backward()
        
        if (step + 1) % grad_accum == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Statistics
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            acc = (pred == y).float().mean()
            
            run["loss"] += (ce.item() + beta * aux.item()) * x.size(0)
            run["ce"] += ce.item() * x.size(0)
            run["aux"] += aux.item() * x.size(0)
            run["acc"] += acc.item() * x.size(0)
            run["n"] += x.size(0)
    
    # Handle remaining gradients
    if run["n"] % (loader.batch_size * grad_accum) != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    if scheduler is not None:
        scheduler.step()
    
    return {
        "loss": run["loss"] / run["n"],
        "ce": run["ce"] / run["n"],
        "aux": run["aux"] / run["n"],
        "acc": run["acc"] / run["n"],
        "lr": scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
    }


@torch.no_grad()
def validate(model, loader, device, use_amp=True):
    """Evaluate top-1 and top-5 accuracy."""
    model.eval()
    ce_loss = nn.CrossEntropyLoss()
    run = {"loss": 0.0, "top1": 0.0, "top5": 0.0, "n": 0}
    
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        
        with autocast(enabled=use_amp):
            logits = model(x)
            loss = ce_loss(logits, y)
        
        # Top-1 accuracy
        pred = logits.argmax(dim=1)
        top1_correct = (pred == y).float().sum()
        
        # Top-5 accuracy
        _, top5_pred = logits.topk(5, dim=1)
        top5_correct = top5_pred.eq(y.view(-1, 1).expand_as(top5_pred)).any(dim=1).float().sum()
        
        run["loss"] += loss.item() * x.size(0)
        run["top1"] += top1_correct.item()
        run["top5"] += top5_correct.item()
        run["n"] += x.size(0)
    
    return {
        "loss": run["loss"] / run["n"],
        "top1": run["top1"] / run["n"],
        "top5": run["top5"] / run["n"],
    }


def print_epoch_summary(phase, epoch, epochs, tr_stats, val_stats, elapsed_s, beta, validated=True):
    """Print epoch summary."""
    aux_term = f"{beta}·Aux {tr_stats['aux']:.4f}" if beta > 0 else "no-aux"
    base_str = (f"[{phase}] Epoch {epoch+1:03d}/{epochs} | "
                f"LR {tr_stats['lr']:.2e} | "
                f"Train Loss {tr_stats['loss']:.4f} (CE {tr_stats['ce']:.4f} + {aux_term}) | "
                f"Train Acc {tr_stats['acc']*100:.2f}%")
    
    if validated:
        print(f"{base_str} | Val Top-1 {val_stats['top1']*100:.2f}% | "
              f"Val Top-5 {val_stats['top5']*100:.2f}% | {elapsed_s:.1f}s")
    else:
        print(f"{base_str} | (val skipped) | {elapsed_s:.1f}s")


def save_checkpoint(model, optimizer, scaler, epoch, val_top1, args, filename):
    """Save training checkpoint."""
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    filepath = os.path.join(args.checkpoint_dir, filename)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'val_top1': val_top1,
        'args': vars(args)
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


# ================== Main ==================

def main():
    args = parse_args()
    
    if not HAS_TIMM:
        raise ImportError("timm is required. Install with: pip install timm")
    
    use_amp = not args.no_amp and torch.cuda.is_available()
    
    # Print configuration
    print("="*80)
    print("ViT-Small on ImageNet-1K with 2:4 Structured Sparsity")
    print("="*80)
    print(f"  Model: {args.model} (pretrained: {args.pretrained})")
    print(f"  Sparsity: {args.sparsity_type} | Reg: {args.reg_type} | Beta: {args.beta}")
    print(f"  Target layers: {args.target_layers}")
    print(f"  Epochs: S1={args.epochs_stage1}, S2={args.epochs_stage2}")
    print(f"  LR: {args.lr} | Batch: {args.batch_size} | AMP: {use_amp}")
    print("="*80 + "\n")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    
    # Load model
    print(f"Loading {args.model}...")
    model = timm.create_model(args.model, pretrained=args.pretrained, num_classes=args.num_classes)
    model = model.to(device)
    
    # Get layers to prune
    layers_to_prune = get_linear_layers_to_prune(model, args.target_layers)
    print(f"Found {len(layers_to_prune)} layers to prune:")
    for name, _ in layers_to_prune[:5]:
        print(f"  - {name}")
    if len(layers_to_prune) > 5:
        print(f"  ... and {len(layers_to_prune) - 5} more")
    print()
    
    # Get data config from model for correct preprocessing
    data_config = timm.data.resolve_model_data_config(model)
    print(f"Model data config:")
    print(f"  Input size: {data_config['input_size']}")
    print(f"  Interpolation: {data_config['interpolation']}")
    print(f"  Crop pct: {data_config.get('crop_pct', 0.875)}")
    print(f"  Mean: {data_config['mean']}")
    print(f"  Std: {data_config['std']}")
    print()
    
    # Data transforms using model's data config
    train_transform = create_transform(
        input_size=data_config['input_size'],
        is_training=True,
        auto_augment='rand-m9-mstd0.5-inc1',
        interpolation=data_config['interpolation'],
        re_prob=0.25,
        re_mode='pixel',
        re_count=1,
        mean=data_config['mean'],
        std=data_config['std'],
    )
    
    val_transform = create_transform(
        input_size=data_config['input_size'],
        is_training=False,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        crop_pct=data_config.get('crop_pct', 0.875),
    )
    
    # Load datasets
    print(f"Loading ImageNet from {args.data_path}...")
    train_dataset = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(args.data_path, 'val'), transform=val_transform)
    print(f"Train: {len(train_dataset):,} | Val: {len(val_dataset):,}\n")
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    # Setup training
    aux_loss_fn = get_aux_loss_fn(args.reg_type)
    scaler = GradScaler(enabled=use_amp)
    
    # Initial evaluation
    print("Initial evaluation...")
    init_val = validate(model, val_loader, device, use_amp)
    print(f"[Init] Val Top-1: {init_val['top1']*100:.2f}% | Top-5: {init_val['top5']*100:.2f}% | "
          f"Loss: {init_val['loss']:.4f}\n")
    
    # Optimizer and Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs_stage1, eta_min=args.lr * 0.01
    )
    
    # ========== Stage 1: Sparse Regularization ==========
    print("=== Stage 1: 2:4 Sparse Regularization ===")
    best_val_top1 = 0.0
    
    for epoch in range(args.epochs_stage1):
        t0 = time()
        tr = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, epoch, 
                            args.epochs_stage1, device, aux_loss_fn, layers_to_prune,
                            args.beta, args.alpha, use_amp, args.grad_accum, args.aux_loss_freq)
        
        if (epoch + 1) % args.val_freq == 0 or epoch == args.epochs_stage1 - 1:
            val = validate(model, val_loader, device, use_amp)
            dt = time() - t0
            print_epoch_summary("S1", epoch, args.epochs_stage1, tr, val, dt, args.beta, True)
            
            if val["top1"] > best_val_top1:
                best_val_top1 = val["top1"]
                save_checkpoint(model, optimizer, scaler, epoch, val["top1"], args,
                              'vit_2_4_s1_best.pth')
        else:
            dt = time() - t0
            print_epoch_summary("S1", epoch, args.epochs_stage1, tr, {}, dt, args.beta, False)
    
    # Apply 2:4 masks
    print(f"\n[Mask] Applying 2:4 structured sparsity...")
    apply_2_4_mask_to_model(model, layers_to_prune)
    
    post_val = validate(model, val_loader, device, use_amp)
    print(f"[PostMask] Val Top-1: {post_val['top1']*100:.2f}% | Top-5: {post_val['top5']*100:.2f}% | "
          f"Loss: {post_val['loss']:.4f}\n")
    
    # ========== Stage 2: Masked Training ==========
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs_stage2, eta_min=args.lr * 0.01
    )
    scaler = GradScaler(enabled=use_amp)
    
    print("=== Stage 2: Masked Fine-tuning ===")
    best_val_top1_s2 = 0.0
    
    for epoch in range(args.epochs_stage2):
        t0 = time()
        tr = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, epoch,
                            args.epochs_stage2, device, aux_loss_fn, layers_to_prune,
                            0.0, args.alpha, use_amp, args.grad_accum, args.aux_loss_freq)  # beta=0 in stage 2
        
        if (epoch + 1) % args.val_freq == 0 or epoch == args.epochs_stage2 - 1:
            val = validate(model, val_loader, device, use_amp)
            dt = time() - t0
            print_epoch_summary("S2", epoch, args.epochs_stage2, tr, val, dt, 0.0, True)
            
            if val["top1"] > best_val_top1_s2:
                best_val_top1_s2 = val["top1"]
                save_checkpoint(model, optimizer, scaler, epoch, val["top1"], args,
                              'vit_2_4_s2_best.pth')
        else:
            dt = time() - t0
            print_epoch_summary("S2", epoch, args.epochs_stage2, tr, {}, dt, 0.0, False)
    
    # Final report
    save_checkpoint(model, optimizer, scaler, args.epochs_stage2-1, best_val_top1_s2, args,
                   'vit_2_4_final.pth')
    sparsity_report(model, layers_to_prune)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print(f"Pretrained: {init_val['top1']*100:.2f}% | S1 Best: {best_val_top1*100:.2f}% | "
          f"PostMask: {post_val['top1']*100:.2f}% | S2 Best: {best_val_top1_s2*100:.2f}%")
    print("="*80)


if __name__ == '__main__':
    main()
