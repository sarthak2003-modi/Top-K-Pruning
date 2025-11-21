# Top-K Pruning (TKP)

This repository contains the implementation code for the paper "Beyond Filter Pruning: Top-K Spatial Selection for Efficient Neural Networks" submitted to CVPR 2026.

## Overview

Top-K Pruning (TKP) is a structured pruning framework that targets spatial redundancy within convolutional kernels. Unlike traditional filter pruning methods that operate at the inter-filter level, TKP extends sparsity to the spatial dimension by retaining only the most informative positions within each convolutional kernel.

### Key Features

- **Two-Stage Training**: Auxiliary regularization followed by deterministic Top-K selection
- **Universal Applicability**: Works across CNNs, Vision Transformers, and Vision-Language Models
- **Hardware-Friendly**: Produces structured sparsity patterns for efficient deployment
- **Quantization-Compatible**: Maintains performance when combined with quantization-aware training

## Repository Structure

```
.
├── vgg8_optimized.py          # VGG-8 on CIFAR-10
├── googlenet_optimized.py     # GoogLeNet on CIFAR-100
├── densenet_optimized.py      # DenseNet-121 on CIFAR-100
└── vit_small.py              # ViT-Small on ImageNet-1K with 2:4 sparsity
```

## Requirements

```bash
pip install torch torchvision timm
```

### Recommended Environment
- Python 3.8+
- PyTorch 2.0+ (for torch.compile support)
- CUDA-capable GPU (optional but recommended)

## Quick Start

### Training VGG-8 on CIFAR-10

```bash
python vgg8_optimized.py \
    --topk 1 \
    --reg-type exponential \
    --beta 3000 \
    --alpha 3.0 \
    --epochs-stage1 100 \
    --epochs-stage2 100 \
    --lr 1e-3 \
    --data-root ./data
```

### Training GoogLeNet on CIFAR-100

```bash
python googlenet_optimized.py \
    --topk 1 \
    --reg-type exponential \
    --beta 3000 \
    --alpha 3.0 \
    --epochs-stage1 100 \
    --epochs-stage2 100 \
    --batch-train 256 \
    --data-root ./data
```

### Training DenseNet-121 on CIFAR-100

```bash
python densenet_optimized.py \
    --topk 1 \
    --reg-type exponential \
    --beta 3000 \
    --alpha 3.0 \
    --epochs-stage1 100 \
    --epochs-stage2 100 \
    --batch-train 256 \
    --data-root ./data
```

### Training ViT-Small on ImageNet-1K (2:4 Sparsity)

```bash
python vit_small.py \
    --model vit_small_patch16_224.augreg_in21k_ft_in1k \
    --data-path /path/to/imagenet \
    --sparsity-type 2:4 \
    --reg-type exponential \
    --beta 1000 \
    --alpha 2.0 \
    --epochs-stage1 30 \
    --epochs-stage2 30 \
    --batch-size 256
```

## Command-Line Arguments

### Common Arguments for CNN Models

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--topk` | int | 1 | Number of weights to keep per kernel (1-9) |
| `--reg-type` | str | exponential | Regularizer type: none, l2, or exponential |
| `--beta` | float | 3000 | Weight on sparsity auxiliary loss |
| `--alpha` | float | 3.0 | Exponential falloff parameter |
| `--epochs-stage1` | int | 100 | Number of epochs for Stage 1 (regularization) |
| `--epochs-stage2` | int | 100 | Number of epochs for Stage 2 (masked fine-tuning) |
| `--lr` | float | 1e-3 | Learning rate |
| `--weight-decay` | float | 1e-4 | Weight decay |
| `--batch-train` | int | 128/256 | Training batch size |
| `--batch-val` | int | 256/512 | Validation batch size |
| `--grad-accum` | int | 1 | Gradient accumulation steps |
| `--val-freq` | int | 5 | Validate every N epochs |
| `--no-amp` | flag | False | Disable mixed precision training |
| `--no-compile` | flag | False | Disable torch.compile |
| `--seed` | int | 42 | Random seed |
| `--data-root` | str | ./data | Path to data directory |
| `--num-workers` | int | 4 | Number of data loading workers |

### ViT-Specific Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | vit_small_... | Model name from timm |
| `--data-path` | str | required | Path to ImageNet dataset |
| `--sparsity-type` | str | 2:4 | Sparsity pattern |
| `--target-layers` | str | qkv,proj,mlp | Which layers to prune |

## Implementation Details

### Two-Stage Training Process

**Stage 1: Auxiliary Regularization**
- Trains the model with exponential auxiliary loss
- Concentrates weight information into dominant spatial positions
- Uses the following loss function:

```
L_total = L_CE + β * L_aux
```

Where L_aux penalizes coefficients beyond top-K with exponentially decaying weights.

**Stage 2: Masked Fine-Tuning**
- Applies hard Top-K mask based on Stage 1 weights
- Masks are applied to both forward and backward passes
- Only selected spatial positions receive gradient updates

### Optimization Features

All scripts include the following optimizations:
- **Mixed Precision Training (AMP)**: Accelerates training with minimal accuracy impact
- **torch.compile**: PyTorch 2.0+ graph optimization for faster execution
- **Gradient Accumulation**: Enables larger effective batch sizes
- **Optimized Data Loading**: Prefetching and persistent workers
- **Reduced Validation Frequency**: Validates every N epochs to save time

### Regularization Types

1. **Exponential (Recommended)**: `--reg-type exponential`
   - Applies exponentially decaying penalties beyond top-K
   - Best for high-sparsity scenarios
   - Default α=3.0 provides strong concentration

2. **L2**: `--reg-type l2`
   - Standard L2 penalty on weights beyond top-K
   - More uniform penalty across positions

3. **None**: `--reg-type none`
   - No auxiliary regularization
   - Directly applies hard mask (not recommended)

## Expected Results

### CIFAR Benchmarks

| Model | Dataset | Top-K | Baseline Acc | Pruned Acc | Speed-Up |
|-------|---------|-------|--------------|------------|----------|
| ResNet-56 | CIFAR-10 | 1 | 93.58% | 91.80% | 8.83× |
| VGG-19 | CIFAR-100 | 1 | 73.50% | 72.77% | 8.90× |
### ImageNet-1K

| Model | Sparsity | Baseline Acc | Pruned Acc | Speed-Up |
|-------|----------|--------------|------------|----------|
| ViT-small | 2:4 | 78.59% | 78.72% | 2.00× |
| ViT-B/16 | 2:4 | 81.07% | 80.93% | 1.78× |

## Dataset Information

### CIFAR-10/100
- **Splits Used**: Training (50,000 images), Test (10,000 images)
- **Note**: Test split is used for validation during training (standard practice)
- **Resolution**: 32×32 → Resized to 224×224 for pretrained models
- **Preprocessing**: Random crop, horizontal flip, normalization

### ImageNet-1K
- **Splits Used**: Train (~1.28M images), Validation (50,000 images)
- **Resolution**: 224×224
- **Preprocessing**: Random resized crop, horizontal flip, center crop for validation


## Tips for Best Performance

1. **Use Mixed Precision**: Keep `--no-amp` flag off for 2× speedup
2. **Enable torch.compile**: Keep `--no-compile` flag off (PyTorch 2.0+)
3. **Adjust Batch Size**: Increase if you have more GPU memory
4. **Gradient Accumulation**: Use `--grad-accum` to simulate larger batches
5. **Validation Frequency**: Set `--val-freq 10` for faster training
6. **Number of Workers**: Set `--num-workers` to match CPU cores

## Troubleshooting

### Out of Memory (OOM) Errors
- Reduce `--batch-train` and `--batch-val`
- Enable gradient accumulation with `--grad-accum 2` or higher
- Reduce `--num-workers`

### Slow Training
- Enable AMP (remove `--no-amp` if present)
- Enable torch.compile (remove `--no-compile` if present)
- Increase `--val-freq` to validate less frequently
- Use SSD for dataset storage

### Poor Accuracy After Pruning
- Try different α values: `--alpha 2.0` or `--alpha 4.0`
- Increase β: `--beta 5000`
- Train for more epochs in Stage 2
- Try L2 regularization: `--reg-type l2`

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{tkp2026,
  title={Beyond Filter Pruning: Top-K Spatial Selection for Efficient Neural Networks},
  author={Anonymous},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

## License

This code is released for research purposes only. Please refer to the LICENSE file for details.

## Acknowledgments

This implementation builds upon:
- PyTorch and torchvision for deep learning framework
- timm library for vision transformer models
- Standard CIFAR and ImageNet datasets
