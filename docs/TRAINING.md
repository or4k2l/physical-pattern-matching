# Training Guide

## Quick Start

Train a model with the baseline configuration:

```bash
python scripts/train.py --config configs/baseline.yaml
```

## Configuration

### Configuration Files

Configuration files are in YAML format. See `configs/` for examples:

- `baseline.yaml` - Standard training with label smoothing
- `margin_loss.yaml` - Training with margin loss for higher confidence

### Configuration Structure

```yaml
model:
  n_classes: 10              # Number of output classes
  features: [64, 128, 256]   # Feature dimensions per layer
  dropout_rate: 0.3          # Dropout rate
  use_residual: true         # Use residual connections

training:
  batch_size: 128
  epochs: 30
  learning_rate: 0.001
  weight_decay: 0.0001
  
  # Loss configuration
  loss_type: "label_smoothing"  # Options: label_smoothing, margin, focal, combined
  label_smoothing: 0.1
  margin: 1.0
  margin_weight: 0.5
  
  # EMA
  ema_enabled: true
  ema_decay: 0.99
  
  # Data
  dataset_name: "cifar10"    # Options: cifar10, cifar100, imagenet2012
  image_size: [32, 32]
  
  # Paths
  checkpoint_dir: "./checkpoints"
  log_dir: "./logs"
```

## Training on Different Datasets

### CIFAR-10

```bash
python scripts/train.py --config configs/baseline.yaml
```

### CIFAR-100

Create a new config with `n_classes: 100` and `dataset_name: "cifar100"`.

### ImageNet

```yaml
model:
  n_classes: 1000
  features: [64, 128, 256, 512]
  
training:
  dataset_name: "imagenet2012"
  image_size: [224, 224]
  batch_size: 64  # Reduce for large images
```

### Custom Dataset

To use a custom dataset, modify `src/robust_vision/data/loaders.py` to add support for your dataset format.

## Training Options

### Resume from Checkpoint

```bash
python scripts/train.py \
  --config configs/baseline.yaml \
  --checkpoint ./checkpoints/checkpoint_15
```

### Custom Experiment Name

```bash
python scripts/train.py \
  --config configs/baseline.yaml \
  --experiment-name my_experiment
```

### Set Random Seed

```bash
python scripts/train.py \
  --config configs/baseline.yaml \
  --seed 1234
```

## Multi-GPU Training

The trainer automatically detects and uses all available GPUs via JAX's `pmap`.

To restrict to specific GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/train.py --config configs/baseline.yaml
```

## Loss Functions

### Label Smoothing Cross-Entropy

Best for general use:

```yaml
loss_type: "label_smoothing"
label_smoothing: 0.1  # 0 = no smoothing, 0.2 = more smoothing
```

### Margin Loss

Encourages confident predictions:

```yaml
loss_type: "margin"
margin: 2.0  # Higher = more separation between classes
```

### Focal Loss

Good for imbalanced datasets:

```yaml
loss_type: "focal"
alpha: 0.25
gamma: 2.0
```

### Combined Loss

Best overall performance:

```yaml
loss_type: "combined"
label_smoothing: 0.1
margin: 2.0
margin_weight: 1.0
```

## Hyperparameter Tuning

### Manual Tuning

Adjust hyperparameters in your config file and retrain.

### Automated Sweep

Run hyperparameter sweep:

```bash
python scripts/hyperparameter_sweep.py \
  --output ./sweep_results \
  --dataset cifar10 \
  --epochs 10
```

This will:
1. Try different combinations of hyperparameters
2. Train models for each configuration
3. Save results to `sweep_results/`
4. Report the best configuration

## Monitoring

### Training Logs

Logs are saved to the `log_dir` specified in config:

```
logs/
├── experiment_name.log      # Training log
└── experiment_name_metrics.jsonl  # Metrics (JSONL format)
```

### Checkpoints

Checkpoints are saved periodically:

```
checkpoints/
├── checkpoint_5
├── checkpoint_10
├── best_checkpoint_18  # Best model by validation accuracy
└── final_checkpoint_30
```

### Visualize Training

```python
from robust_vision.evaluation.visualization import plot_training_history

plot_training_history(
    "logs/experiment_metrics.jsonl",
    output_path="training_curves.png"
)
```

## Best Practices

### 1. Start with Baseline

Always start with `configs/baseline.yaml` and adjust from there.

### 2. Use EMA

Always enable EMA for better generalization:

```yaml
ema_enabled: true
ema_decay: 0.99
```

### 3. Label Smoothing

Use 0.1 for most tasks:

```yaml
label_smoothing: 0.1
```

### 4. Learning Rate

Start with 1e-3 and adjust based on convergence:

- Too high: Loss oscillates or increases
- Too low: Very slow convergence

### 5. Batch Size

- Larger batch size: Faster training, more stable gradients
- Smaller batch size: Better generalization, less memory

Rule of thumb: As large as fits in memory.

### 6. Regularization

For overfitting, try:

```yaml
dropout_rate: 0.4  # Increase dropout
weight_decay: 0.001  # Increase weight decay
label_smoothing: 0.2  # Increase smoothing
```

## Troubleshooting

### Training is Too Slow

1. Check GPU utilization: `nvidia-smi`
2. Increase batch size if possible
3. Enable prefetching (already enabled by default)
4. Use smaller model for debugging

### Loss is NaN

1. Reduce learning rate
2. Check for data issues (NaNs, extreme values)
3. Use gradient clipping

### Accuracy is Not Improving

1. Visualize training curves
2. Check if model is learning (loss should decrease)
3. Try different learning rate
4. Add more regularization if overfitting

## Next Steps

- Run robustness evaluation: [Evaluation Guide](#evaluation)
- Deploy your model: [DEPLOYMENT.md](DEPLOYMENT.md)
- Try different architectures
