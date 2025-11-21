# %% =============================== #
#  GoogLeNet (Inception v1) on CIFAR-100 @ 224x224 - OPTIMIZED VERSION
#  Stage-1: CE + β·AuxReg (top-k per-kernel on ALL Conv2d)
#  -> Hard+Grad Mask (top-k per-kernel) -> Stage-2 train
#  
#  Optimizations:
#  - Mixed precision training (AMP)
#  - Optimized data loading (prefetch, larger batches)
#  - Gradient accumulation
#  - torch.compile (PyTorch 2.0+)
#  - Efficient auxiliary loss computation
#  - Reduced validation frequency
# %% =============================== #
import argparse
import math
import random
from time import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


def parse_args():
    parser = argparse.ArgumentParser(
        description='GoogLeNet sparse training on CIFAR-100 (optimized)'
    )
    
    # Sparsity configuration
    parser.add_argument('--topk', type=int, default=1, choices=range(1, 10),
                        help='Number of weights to keep per kernel (1-9, default: 1)')
    parser.add_argument('--reg-type', type=str, default='exponential',
                        choices=['none', 'l2', 'exponential'],
                        help='Regularizer type: none, l2, or exponential (default: exponential)')
    parser.add_argument('--beta', type=float, default=3000,
                        help='Weight on sparsity auxiliary loss (default: 3000)')
    parser.add_argument('--alpha', type=float, default=3.0,
                        help='Exponential falloff parameter (only for exponential reg, default: 3.0)')
    
    # Training configuration
    parser.add_argument('--epochs-stage1', type=int, default=100,
                        help='Number of epochs for stage 1 (default: 100)')
    parser.add_argument('--epochs-stage2', type=int, default=100,
                        help='Number of epochs for stage 2 (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay (default: 1e-4)')
    parser.add_argument('--batch-train', type=int, default=256,
                        help='Training batch size (default: 256, increased for speed)')
    parser.add_argument('--batch-val', type=int, default=512,
                        help='Validation batch size (default: 512)')
    parser.add_argument('--grad-accum', type=int, default=1,
                        help='Gradient accumulation steps (default: 1)')
    
    # Optimization settings
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable mixed precision training')
    parser.add_argument('--no-compile', action='store_true',
                        help='Disable torch.compile')
    parser.add_argument('--val-freq', type=int, default=5,
                        help='Validate every N epochs (default: 5, set to 1 for every epoch)')
    parser.add_argument('--aux-loss-freq', type=int, default=1,
                        help='Compute aux loss every N steps (default: 1)')
    
    # Other settings
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--data-root', type=str, default='./data',
                        help='Path to data directory (default: ./data)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers (default: 4, increased)')
    parser.add_argument('--label-smooth', type=float, default=0.0,
                        help='Label smoothing (default: 0.0)')
    
    return parser.parse_args()


# ----------------- Optimized Aux loss functions -----------------
def sparsity_aux_loss_none(model: nn.Module, k: int = 1, alpha: float = 0.3):
    """No auxiliary regularization - returns zero."""
    return torch.tensor(0.0, device=next(model.parameters()).device)


def sparsity_aux_loss_l2_beyond_topk(model: nn.Module, k: int = 1, alpha: float = 0.3):
    """
    L2 regularization on weights beyond top-k (optimized version).
    Uses efficient tensor operations and minimizes memory allocations.
    """
    total_penalty = 0.0
    total_patches = 0
    
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            W = m.weight
            cout, cin_g, kh, kw = W.shape
            N = cout * cin_g
            kernel_size = kh * kw
            k_eff = min(k, kernel_size)
            
            if k_eff < kernel_size:
                # Flatten and get absolute values
                flat = W.view(N, kernel_size)
                flat_sq = flat * flat  # More efficient than .abs() then square
                
                # Use topk to get indices to keep, then mask everything else
                _, topk_idx = torch.topk(flat.abs(), k=k_eff, dim=1, largest=True, sorted=False)
                
                # Create mask (0 for top-k, 1 for rest)
                mask = torch.ones_like(flat_sq)
                mask.scatter_(1, topk_idx, 0.0)
                
                # Sum penalties for weights beyond top-k
                layer_penalty = (flat_sq * mask).sum()
                total_penalty += layer_penalty
            
            total_patches += N
    
    if total_patches == 0:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    return total_penalty / total_patches


def sparsity_aux_loss_exponential_topk(model: nn.Module, k: int = 1, alpha: float = 0.3):
    """
    Exponential falloff regularization (optimized version).
    Pre-computes exponential weights and uses efficient operations.
    """
    total_penalty = 0.0
    total_patches = 0
    
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            W = m.weight
            cout, cin_g, kh, kw = W.shape
            N = cout * cin_g
            kernel_size = kh * kw
            k_eff = min(k, kernel_size)
            
            if k_eff < kernel_size:
                # Flatten and compute
                flat = W.view(N, kernel_size)
                flat_sq = flat * flat
                
                # Sort by absolute value
                vals_sq, indices = torch.sort(flat_sq, dim=1, descending=True)
                
                # Pre-compute exponential weights for positions beyond k
                ranks = torch.arange(kernel_size, device=W.device)
                exp_weights = torch.where(
                    ranks >= k_eff,
                    torch.exp(-alpha * (ranks - k_eff).float()),
                    torch.zeros_like(ranks, dtype=W.dtype)
                )
                
                # Apply weights (broadcasting)
                layer_penalty = (vals_sq * exp_weights.unsqueeze(0)).sum()
                total_penalty += layer_penalty
            
            total_patches += N
    
    if total_patches == 0:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    return total_penalty / total_patches


def get_aux_loss_fn(reg_type: str):
    """Return the appropriate auxiliary loss function based on regularizer type."""
    if reg_type == 'none':
        return sparsity_aux_loss_none
    elif reg_type == 'l2':
        return sparsity_aux_loss_l2_beyond_topk
    elif reg_type == 'exponential':
        return sparsity_aux_loss_exponential_topk
    else:
        raise ValueError(f"Unknown regularizer type: {reg_type}")


# ----------------- Utils -----------------
def cosine_lr(optimizer, base_lr, epoch_idx, total_epochs, min_lr_mult=0.0):
    if total_epochs <= 1:
        lr = base_lr
    else:
        cos = 0.5 * (1 + math.cos(math.pi * epoch_idx / (total_epochs - 1)))
        lr = min_lr_mult * base_lr + (1 - min_lr_mult) * base_lr * cos
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    return lr


@torch.no_grad()
def eval_top1(model: nn.Module, loader: DataLoader, device, label_smooth=0.0, use_amp=True):
    """Optimized evaluation with AMP support."""
    model.eval()
    ce = nn.CrossEntropyLoss(label_smoothing=label_smooth)
    correct, total, loss_sum = 0, 0, 0.0
    
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        
        with autocast(enabled=use_amp):
            logits = model(x)
            loss = ce(logits, y)
        
        loss_sum += loss.item() * y.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    
    return {"top1": correct / total, "loss": loss_sum / max(1, total)}


# ----------------- Optimized mask operations -----------------
@torch.no_grad()
def make_topk_mask(weight: torch.Tensor, k: int):
    """Optimized top-k mask creation."""
    cout, cin_g, kh, kw = weight.shape
    kernel_size = kh * kw
    k_eff = min(k, kernel_size)
    
    # Reshape and compute in one go
    flat = weight.view(cout * cin_g, kernel_size).abs()
    
    # Use topk directly (more efficient)
    topk_idx = torch.topk(flat, k=k_eff, dim=1, largest=True, sorted=False).indices
    
    # Create mask efficiently
    mask_flat = torch.zeros_like(flat, dtype=weight.dtype)
    mask_flat.scatter_(1, topk_idx, 1.0)
    
    return mask_flat.view_as(weight)


def apply_hard_and_grad_mask_all_convs(model: nn.Module, k: int = 1):
    """Optimized masking with minimal overhead."""
    total_kept, total_elems = 0, 0
    
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            W = m.weight.data
            mask = make_topk_mask(W, k).to(W.device)
            
            # Apply mask
            m.weight.data.mul_(mask)
            
            # Count
            kept_here = mask.sum().item()
            total_kept += kept_here
            total_elems += W.numel()
            
            # Store mask & register hook (optimized)
            if hasattr(m, "weight_mask"):
                m.weight_mask.copy_(mask)
            else:
                m.register_buffer("weight_mask", mask)
            
            # Register hook with closure
            def _make_mask_grad(mask_ref):
                def _hook(grad):
                    return grad * mask_ref
                return _hook
            
            m.weight.register_hook(_make_mask_grad(m.weight_mask))
    
    density = 100.0 * total_kept / max(1, total_elems)
    print(f"[Mask] Applied TOP-{k} to ALL Conv2d kernels: kept {total_kept}/{total_elems} "
          f"({density:.2f}% density).")


# ----------------- BN recalibration (optimized) -----------------
@torch.no_grad()
def bn_recalibrate(model, loader, device, num_batches=100, use_amp=True):
    """Optimized BN recalibration with fewer batches and AMP."""
    model.train()
    saved = {}
    
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            saved[m] = m.momentum
            m.reset_running_stats()
            m.momentum = None
    
    seen = 0
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        with autocast(enabled=use_amp):
            # GoogLeNet returns aux outputs during training, just use main output
            outputs = model(x)
            if isinstance(outputs, tuple):
                _ = outputs[0]  # main output
            else:
                _ = outputs
        seen += 1
        if seen >= num_batches:
            break
    
    for m, mom in saved.items():
        m.momentum = mom
    
    model.eval()


# ----------------- Sparsity report (optimized) -----------------
@torch.no_grad()
def sparsity_report_all_convs(model: nn.Module, list_top=12):
    """Optimized sparsity reporting."""
    total_nz, total = 0, 0
    rows = []
    
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            W = m.weight.data
            nz = (W != 0).sum().item()
            n = W.numel()
            total_nz += nz
            total += n
            rows.append((name, m.kernel_size, m.groups, nz, n, 100.0 * nz / n))
    
    if total == 0:
        print("[Sparsity] No Conv2d layers found.")
        return
    
    density = 100.0 * total_nz / total
    print("\n========== Final Sparsity Report (ALL Conv2d) ==========")
    print(f"Overall: non-zeros {total_nz}/{total} -> Density {density:.2f}% | "
          f"Sparsity {100.0 - density:.2f}%")
    
    rows.sort(key=lambda r: r[4], reverse=True)
    print("Per-layer (largest first): name | ksize | groups | nz/total | density | sparsity")
    for name, ksz, g, nz, n, dens in rows[:list_top]:
        print(f"  - {name:40s} {str(ksz):>5s}  g={g:<2d}  {nz:9d}/{n:9d}  "
              f"{dens:6.2f}%  {100.0 - dens:6.2f}%")
    print("========================================================\n")


# ----------------- Optimized training loop -----------------
def train_one_epoch(model, loader, optimizer, scaler, epoch, total_epochs, device, 
                    aux_loss_fn, beta, alpha, label_smooth, base_lr, use_amp, 
                    grad_accum_steps=1, aux_loss_freq=1):
    """Optimized training loop with AMP and gradient accumulation."""
    model.train()
    ce = nn.CrossEntropyLoss(label_smoothing=label_smooth)
    run = {"loss":0.0, "ce":0.0, "aux":0.0, "acc":0.0, "n":0}
    lr_now = cosine_lr(optimizer, base_lr, epoch_idx=epoch, total_epochs=total_epochs, min_lr_mult=0.0)
    
    optimizer.zero_grad(set_to_none=True)
    
    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        
        with autocast(enabled=use_amp):
            # GoogLeNet returns (main_output, aux1, aux2) during training
            # We only use main output for our sparse training
            outputs = model(x)
            if isinstance(outputs, tuple):
                logits = outputs[0]  # main output
            else:
                logits = outputs
            
            loss_ce = ce(logits, y)
            
            # Compute aux loss less frequently if specified
            if batch_idx % aux_loss_freq == 0:
                loss_aux = aux_loss_fn(model, k=args.topk, alpha=alpha)
            else:
                loss_aux = torch.tensor(0.0, device=device)
            
            loss = loss_ce + beta * loss_aux
            loss = loss / grad_accum_steps  # Scale for gradient accumulation
        
        # Backward with gradient scaling
        scaler.scale(loss).backward()
        
        # Update weights every grad_accum_steps
        if (batch_idx + 1) % grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        # Tracking
        bs = y.size(0)
        run["loss"] += loss.item() * bs * grad_accum_steps
        run["ce"] += loss_ce.item() * bs
        run["aux"] += loss_aux.item() * bs
        run["acc"] += (logits.argmax(1) == y).float().sum().item()
        run["n"] += bs
    
    return {
        "lr": lr_now,
        "loss": run["loss"] / run["n"],
        "ce": run["ce"] / run["n"],
        "aux": run["aux"] / run["n"],
        "acc": run["acc"] / run["n"],
    }


def print_epoch_summary(phase, epoch, epochs, tr_stats, val_acc, val_loss, elapsed_s, beta, validated=True):
    """Print epoch summary with optional validation."""
    aux_term = f"{beta}·Aux {tr_stats['aux']:.4f}" if beta > 0 else "no-aux"
    base_str = (f"[{phase}] Epoch {epoch+1:03d}/{epochs} | "
                f"LR {tr_stats['lr']:.2e} | "
                f"Train Loss {tr_stats['loss']:.4f} (CE {tr_stats['ce']:.4f} + {aux_term}) | "
                f"Train Acc {tr_stats['acc']*100:.2f}%")
    
    if validated:
        print(f"{base_str} | Val Acc {val_acc*100:.2f}% | Val Loss {val_loss:.4f} | {elapsed_s:.1f}s")
    else:
        print(f"{base_str} | (val skipped) | {elapsed_s:.1f}s")


# ----------------- Model loading -----------------
def load_googlenet_cifar100(pretrained=True):
    """Load GoogLeNet (Inception v1) and adapt for CIFAR-100."""
    weights = models.GoogLeNet_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.googlenet(weights=weights)
    
    # Replace final FC layer for CIFAR-100 (100 classes)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, 100)
    
    # GoogLeNet has auxiliary classifiers (aux1, aux2) used during training
    # We'll keep them but only use main output for our sparse training
    # The auxiliary classifiers help with gradient flow in deep networks
    if hasattr(model, 'aux1') and model.aux1 is not None:
        model.aux1.fc2 = nn.Linear(model.aux1.fc2.in_features, 100)
    if hasattr(model, 'aux2') and model.aux2 is not None:
        model.aux2.fc2 = nn.Linear(model.aux2.fc2.in_features, 100)
    
    # Disable auxiliary classifiers during inference (standard practice)
    model.aux_logits = False
    
    return model


# ----------------- Main -----------------
if __name__ == '__main__':
    args = parse_args()
    
    # Determine if AMP should be used
    use_amp = not args.no_amp and torch.cuda.is_available()
    use_compile = not args.no_compile and hasattr(torch, 'compile')
    
    # Print configuration
    print("="*70)
    print("Configuration:")
    print(f"  Model: GoogLeNet (Inception v1)")
    print(f"  Top-k per kernel: {args.topk}")
    print(f"  Regularizer type: {args.reg_type}")
    print(f"  Beta (aux weight): {args.beta}")
    if args.reg_type == 'exponential':
        print(f"  Alpha (exp falloff): {args.alpha}")
    print(f"  Stage 1 epochs: {args.epochs_stage1}")
    print(f"  Stage 2 epochs: {args.epochs_stage2}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Batch size (train/val): {args.batch_train}/{args.batch_val}")
    print(f"  Gradient accumulation: {args.grad_accum} steps")
    print(f"  Validation frequency: every {args.val_freq} epochs")
    print(f"  Aux loss frequency: every {args.aux_loss_freq} steps")
    print(f"  Random seed: {args.seed}")
    print(f"  Mixed precision (AMP): {use_amp}")
    print(f"  torch.compile: {use_compile}")
    print(f"  Num workers: {args.num_workers}")
    print("="*70 + "\n")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    
    # Data (CIFAR-100 @ 224x224, ImageNet stats)
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    
    # Optimized transforms
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    
    train_set = datasets.CIFAR100(root=args.data_root, train=True, download=True, transform=train_tfms)
    val_set = datasets.CIFAR100(root=args.data_root, train=False, download=True, transform=val_tfms)
    
    # Optimized data loaders with prefetch and persistent workers
    train_loader = DataLoader(
        train_set, batch_size=args.batch_train, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        drop_last=True, prefetch_factor=2 if args.num_workers > 0 else None
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_val, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    
    # Model
    print("Loading model...")
    model = load_googlenet_cifar100(pretrained=True).to(device)
    
    # Compile model if available (PyTorch 2.0+)
    if use_compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model, mode='default')
    
    # Get auxiliary loss function
    aux_loss_fn = get_aux_loss_fn(args.reg_type)
    
    # Gradient scaler for AMP
    scaler = GradScaler(enabled=use_amp)
    
    # Initial evaluation
    print("Running initial evaluation...")
    init = eval_top1(model, val_loader, device, args.label_smooth, use_amp)
    print(f"[Init] Val@1 {init['top1']*100:.2f}% | Val Loss {init['loss']:.4f}\n")
    
    # ----------------- Stage 1 -----------------
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    
    print(f"=== Stage 1: Training with {args.reg_type} regularization ===")
    for epoch in range(args.epochs_stage1):
        t0 = time()
        tr = train_one_epoch(
            model, train_loader, optimizer, scaler, epoch, args.epochs_stage1,
            device, aux_loss_fn, args.beta, args.alpha, args.label_smooth, args.lr,
            use_amp, args.grad_accum, args.aux_loss_freq
        )
        
        # Validate only every val_freq epochs (and on last epoch)
        if (epoch + 1) % args.val_freq == 0 or epoch == args.epochs_stage1 - 1:
            val = eval_top1(model, val_loader, device, args.label_smooth, use_amp)
            dt = time() - t0
            print_epoch_summary("S1", epoch, args.epochs_stage1, tr, val["top1"], val["loss"], dt, args.beta, True)
        else:
            dt = time() - t0
            print_epoch_summary("S1", epoch, args.epochs_stage1, tr, 0, 0, dt, args.beta, False)
    
    # ----------------- Apply hard+grad mask + BN recal -----------------
    print(f"\n[Mask] Applying hard+grad mask (top-{args.topk} per kernel) to ALL Conv2d...")
    apply_hard_and_grad_mask_all_convs(model, k=args.topk)
    
    print("[BN] Recalibrating BN running stats (with fewer batches)...")
    bn_recalibrate(model, train_loader, device, num_batches=100, use_amp=use_amp)
    
    post = eval_top1(model, val_loader, device, args.label_smooth, use_amp)
    print(f"[PostMask] Val@1 {post['top1']*100:.2f}% | Val Loss {post['loss']:.4f}\n")
    
    # ----------------- Stage 2 -----------------
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    scaler = GradScaler(enabled=use_amp)  # Reset scaler
    
    print("=== Stage 2: Training with hard+grad mask active ===")
    for epoch in range(args.epochs_stage2):
        t0 = time()
        tr = train_one_epoch(
            model, train_loader, optimizer, scaler, epoch, args.epochs_stage2,
            device, aux_loss_fn, args.beta, args.alpha, args.label_smooth, args.lr,
            use_amp, args.grad_accum, args.aux_loss_freq
        )
        
        # Validate only every val_freq epochs (and on last epoch)
        if (epoch + 1) % args.val_freq == 0 or epoch == args.epochs_stage2 - 1:
            val = eval_top1(model, val_loader, device, args.label_smooth, use_amp)
            dt = time() - t0
            print_epoch_summary("S2", epoch, args.epochs_stage2, tr, val["top1"], val["loss"], dt, args.beta, True)
        else:
            dt = time() - t0
            print_epoch_summary("S2", epoch, args.epochs_stage2, tr, 0, 0, dt, args.beta, False)
    
    # ----------------- Final sparsity report -----------------
    sparsity_report_all_convs(model)
    print("Done.")
