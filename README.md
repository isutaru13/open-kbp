# OpenKBP Dose Prediction with MONAI

![](read-me-images/aapm.png)

A MONAI-based 3D U-Net implementation for predicting radiation dose distributions in head-and-neck cancer patients. This is a refactored and **optimized** version of the [OpenKBP Grand Challenge](https://aapm.onlinelibrary.wiley.com/doi/epdf/10.1002/mp.14845) codebase, using PyTorch and MONAI instead of TensorFlow/Keras.

**Includes two model architectures:**
- **MONAI 3D U-Net** - Standard residual U-Net
- **HD U-Net (NEW)** - Hierarchical Dense U-Net with attention gates and deep supervision

**Optimized for RTX 3060 12GB** with Mixed Precision, Gradient Accumulation, OneCycleLR, and more.

![](read-me-images/pipeline.png)

## Citation

If you use this dataset or code, please cite the original paper:

> A. Babier, B. Zhang, R. Mahmood, K.L. Moore, T.G. Purdie, A.L. McNiven, T.C.Y. Chan, "[OpenKBP: The open-access knowledge-based planning grand challenge and dataset](https://aapm.onlinelibrary.wiley.com/doi/epdf/10.1002/mp.14845)," _Medical Physics_, Vol. 48, pp. 5549-5561, 2021.

## Table of Contents

- [Features](#features)
- [Data](#data)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [HD U-Net Training](#hd-u-net-training)
- [RTX 3060 Optimization Guide](#rtx-3060-optimization-guide)
- [Command Line Reference](#command-line-reference)
- [GPU Configuration](#gpu-configuration)
- [Evaluation Metrics](#evaluation-metrics)
- [Output Files](#output-files)
- [Legacy Code](#legacy-code)
- [Competition Results](#competition-results)

## Features

### Core Features
- **MONAI 3D U-Net**: Modern PyTorch-based architecture with residual units
- **HD U-Net (NEW)**: Hierarchical Dense U-Net with attention gates and deep supervision
- **Modular Design**: Clean separation of dataset, model, transforms, losses, and evaluation
- **Comprehensive Evaluation**: Dose score (MAE) and DVH metrics
- **Result Exports**: JSON summaries, CSV losses, and visualization plots

### HD U-Net Features (NEW)
- **Dense Blocks**: DenseNet-style feature reuse with bottleneck design
- **Attention Gates**: Enhanced skip connections that focus on relevant features
- **Deep Supervision**: Auxiliary losses for improved gradient flow
- **Three Variants**: lite (~200K params), standard (~2M params), large (~8M params)
- **Gradient Checkpointing**: Memory-efficient training for larger models

### Optimization Features (NEW)
- **Mixed Precision (AMP)**: ~1.5-2× speedup with FP16 training
- **Gradient Accumulation**: Larger effective batch sizes without more VRAM
- **OneCycleLR Scheduler**: Faster convergence in fewer epochs
- **Cosine Annealing**: Alternative smooth LR decay
- **Gradient Clipping**: Training stability for long runs
- **AdamW Optimizer**: Better weight decay handling
- **Persistent Workers**: Faster data loading between epochs
- **torch.compile Support**: PyTorch 2.0+ optimization
- **VRAM Monitoring**: Real-time memory usage tracking
- **Warmup Period**: Avoid saving premature "best" models

## Data

The dataset contains 340 head-and-neck cancer patients treated with intensity modulated radiation therapy (IMRT):

| Set | Patients | Patient IDs |
|-----|----------|-------------|
| Training | 200 | pt_1 - pt_200 |
| Validation | 40 | pt_201 - pt_240 |
| Testing | 100 | pt_241 - pt_340 |

Each patient folder contains:

| File | Description |
|------|-------------|
| `ct.csv` | CT scan (sparse format) |
| `dose.csv` | Reference dose distribution (sparse format) |
| `possible_dose_mask.csv` | Mask where dose can be non-zero |
| `voxel_dimensions.csv` | Physical voxel size (mm) |
| `{ROI}.csv` | Structure masks for organs and targets |

**Regions of Interest (ROIs):**
- **Organs at Risk**: Brainstem, SpinalCord, RightParotid, LeftParotid, Esophagus, Larynx, Mandible
- **Targets**: PTV56, PTV63, PTV70

## Project Structure

```
open-kbp/
├── src/                          # Main source package
│   ├── __init__.py               # Public API exports
│   ├── constants.py              # Project constants (ROIs, shapes, config)
│   ├── data_utils.py             # Sparse/dense data conversion
│   ├── dataset.py                # OpenKBPDataset PyTorch class
│   ├── evaluation.py             # Dose and DVH metric computation
│   ├── export.py                 # JSON/CSV result exports
│   ├── losses.py                 # Custom masked loss functions
│   ├── model.py                  # MONAI U-Net trainer (optimized)
│   ├── hd_unet.py                # HD U-Net architecture (NEW)
│   ├── hd_unet_model.py          # HD U-Net trainer (NEW)
│   ├── transforms.py             # MONAI data transforms
│   └── visualization.py          # Training plots and dose visualization
├── provided-data/                # Patient data
│   ├── train-pats/               # 200 training patients
│   ├── validation-pats/          # 40 validation patients
│   └── test-pats/                # 100 test patients
├── legacy/                       # Original TensorFlow/Keras code
├── results/                      # Training outputs (gitignored)
├── train_monai.py                # MONAI U-Net training script
├── train_hd_unet.py              # HD U-Net training script (NEW)
├── requirements.txt              # Python dependencies
└── README.md
```

## Installation

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA (RTX 3060 12GB or better recommended)
- PyTorch 2.0+ (for torch.compile support)

### Setup

```bash
# Clone repository
git clone https://github.com/ababier/open-kbp
cd open-kbp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- PyTorch >= 2.0.0
- MONAI >= 1.3.0
- NumPy, Pandas, Matplotlib, tqdm

## Quick Start

### Verify Installation (5 minutes)

```bash
python train_monai.py --data-fraction 0.1 --epochs 5 --batch-size 4
```

### Basic Training

```bash
# Default settings (100 epochs)
python train_monai.py --epochs 100

# With recommended optimizations
python train_monai.py --epochs 100 --batch-size 4 --filters 64 --scheduler onecycle
```

## HD U-Net Training

The HD U-Net (Hierarchical Dense U-Net) offers enhanced feature extraction through dense blocks, attention gates, and deep supervision.

### Model Variants

| Variant | Parameters | VRAM (batch=2) | Recommended For |
|---------|------------|----------------|------------------|
| **lite** | ~200K | ~5 GB | Quick experiments, memory-constrained |
| **standard** | ~2M | ~8 GB | Balanced quality/speed |
| **large** | ~8M | ~11 GB | Best quality, more VRAM needed |

### Quick Test

```bash
python train_hd_unet.py --data-fraction 0.1 --epochs 5 --variant lite
```

### Recommended Training (RTX 3060 12GB)

```bash
# Lite variant - fits comfortably, ~13 hours for 300 epochs
python train_hd_unet.py \
    --epochs 300 \
    --variant lite \
    --batch-size 2 \
    --grad-accum 8 \
    --lr 2e-4 \
    --scheduler onecycle \
    --warmup-epochs 30 \
    --save-freq 50 \
    --grad-checkpoint
```

### Background Training

```bash
# Run in background with logs
nohup python train_hd_unet.py \
    --epochs 300 \
    --variant lite \
    --batch-size 2 \
    --grad-accum 8 \
    --lr 2e-4 \
    --scheduler onecycle \
    --warmup-epochs 30 \
    --save-freq 50 \
    --grad-checkpoint \
    > training.log 2>&1 &

# Monitor progress
tail -f training.log
```

### HD U-Net Specific Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--variant` | `standard` | Model variant: lite, standard, large |
| `--init-features` | `48` | Initial features (custom config) |
| `--growth-rate` | `16` | Dense block growth rate |
| `--no-attention` | - | Disable attention gates |
| `--no-deep-supervision` | - | Disable deep supervision |
| `--dropout` | `0.2` | Dropout rate |

## RTX 3060 Optimization Guide

This codebase is specifically optimized for **RTX 3060 12GB** with a **20-hour training budget**.

### Performance Summary

| Optimization | Speedup | VRAM Reduction |
|--------------|---------|----------------|
| Mixed Precision (AMP) | ~1.5-2× | ~40% |
| Gradient Accumulation | - | Enables larger effective batch |
| OneCycleLR | Fewer epochs needed | - |
| Persistent Workers | ~10-15% | - |

### Time Budget Analysis

With AMP enabled on RTX 3060:

| Config | Epoch Time | 20 Hours = | Recommended |
|--------|------------|------------|-------------|
| batch=4, filters=64 | ~25s | ~2,880 epochs | ✅ |
| batch=4, filters=96 | ~35s | ~2,050 epochs | ⚠️ VRAM tight |
| batch=2, filters=96 | ~30s | ~2,400 epochs | ✅ |

### Recommended Commands

#### Option 1: Balanced (RECOMMENDED)

Best balance of model capacity and training time. **~2.8 hours**.

```bash
python train_monai.py \
    --epochs 400 \
    --batch-size 4 \
    --filters 64 \
    --lr 3e-4 \
    --scheduler onecycle \
    --grad-accum 4 \
    --warmup-epochs 20 \
    --save-freq 50
```

#### Option 2: Maximum Training (Full 20 Hours)

Use the entire time budget for maximum convergence. **~7 hours**.

```bash
python train_monai.py \
    --epochs 1000 \
    --batch-size 4 \
    --filters 64 \
    --lr 2e-4 \
    --scheduler onecycle \
    --grad-accum 4 \
    --warmup-epochs 50 \
    --save-freq 100
```

#### Option 3: Larger Model

Larger model capacity with fewer epochs. **~2.5 hours**.

```bash
python train_monai.py \
    --epochs 300 \
    --batch-size 2 \
    --filters 96 \
    --lr 3e-4 \
    --scheduler onecycle \
    --grad-accum 8 \
    --warmup-epochs 15 \
    --save-freq 50
```

#### Option 4: Quick Experiment

Fast iteration for hyperparameter testing. **~30 minutes**.

```bash
python train_monai.py \
    --epochs 50 \
    --batch-size 4 \
    --filters 64 \
    --lr 3e-4 \
    --scheduler onecycle \
    --grad-accum 4 \
    --data-fraction 0.5
```

#### Option 5: No Augmentation (Recommended First Try)

Test without geometric augmentation - may significantly improve results.

```bash
python train_monai.py \
    --epochs 400 \
    --batch-size 4 \
    --filters 64 \
    --lr 3e-4 \
    --scheduler onecycle \
    --grad-accum 4 \
    --warmup-epochs 20 \
    --augment-type none
```

### Augmentation Strategy

> ⚠️ **Important**: Geometric augmentation (flip/rotate) may **hurt** dose prediction results!

#### Why Geometric Augmentation Can Be Harmful

Unlike segmentation tasks, dose prediction has a **hidden dependency on beam geometry**:

```
Input to model:              What actually determines dose:
┌─────────────────┐         ┌─────────────────────────────┐
│ CT scan         │         │ CT scan                     │
│ + ROI masks     │   →     │ + ROI masks                 │
└─────────────────┘         │ + BEAM ANGLES (not in input)│
                            └─────────────────────────────┘
```

When you flip or rotate the anatomy:
- The dose pattern should change (beams enter from different angles)
- But we flip/rotate the dose too, creating **inconsistent training pairs**
- The model learns a "blurry average" instead of precise mappings

#### Augmentation Types

| Type | Transforms | Safe for Dose? | Use Case |
|------|------------|----------------|----------|
| `none` | No augmentation | ✅ Safest | Try this first |
| `intensity` | Noise, contrast, shift | ✅ Safe | Recommended default |
| `geometric` | Flip, rotate | ⚠️ May hurt | Not recommended |
| `full` | All of the above | ⚠️ May hurt | Use with caution |

#### Recommendation

1. **Start with `--augment-type none`** to establish baseline
2. **Try `--augment-type intensity`** (adds noise/contrast only)
3. **Compare results** - if geometric hurts, avoid it

### Optimization Flags Explained

| Flag | What It Does | When to Use |
|------|--------------|-------------|
| `--amp` | FP16 mixed precision (ON by default) | Always |
| `--no-amp` | Disable AMP, use FP32 | Debugging numerical issues |
| `--grad-accum N` | Accumulate gradients over N batches | When VRAM limited |
| `--scheduler onecycle` | OneCycleLR (fast convergence) | Long training runs |
| `--scheduler cosine` | Cosine annealing | Alternative to onecycle |
| `--scheduler plateau` | ReduceLROnPlateau | Short runs, uncertain epochs |
| `--warmup-epochs N` | Don't save "best" model for N epochs | Avoid early anomalies |
| `--grad-clip N` | Clip gradients to norm N | Training stability |
| `--compile` | Use torch.compile | PyTorch 2.0+, experimental |
| `--augment-type` | Augmentation strategy | See section above |

### Effective Batch Size

The **effective batch size** = `batch-size` × `grad-accum`

| batch-size | grad-accum | Effective Batch | VRAM Usage |
|------------|------------|-----------------|------------|
| 4 | 1 | 4 | ~8 GB |
| 4 | 4 | 16 | ~8 GB |
| 4 | 8 | 32 | ~8 GB |
| 2 | 8 | 16 | ~5 GB |
| 8 | 2 | 16 | ~11 GB |

**Recommendation**: Use `--batch-size 4 --grad-accum 4` for effective batch of 16.

## Command Line Reference

### Model Settings

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-name` | `monai_unet` | Name for the model |
| `--filters` | `32` | Initial U-Net filters (32/64/96/128) |

### Training Settings

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | `100` | Number of training epochs |
| `--batch-size` | `2` | Batch size per GPU |
| `--lr` | `1e-4` | Learning rate |
| `--lr-patience` | `5` | Patience for plateau scheduler |
| `--warmup-epochs` | `10` | Epochs before saving best model |
| `--min-epochs` | `0` | Minimum epochs before early stopping |

### Optimization Settings

| Argument | Default | Description |
|----------|---------|-------------|
| `--amp` | `True` | Enable mixed precision (FP16) |
| `--no-amp` | - | Disable AMP |
| `--grad-accum` | `1` | Gradient accumulation steps |
| `--scheduler` | `onecycle` | LR scheduler (plateau/onecycle/cosine) |
| `--grad-checkpoint` | `False` | Enable gradient checkpointing |
| `--compile` | `False` | Use torch.compile |
| `--grad-clip` | `1.0` | Gradient clipping norm (0 to disable) |

### Data Settings

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-fraction` | `1.0` | Fraction of data to use |
| `--test-split` | `0.1` | Hold-out test split |
| `--num-workers` | `4` | DataLoader workers |
| `--prefetch` | `2` | Prefetch factor |
| `--persistent-workers` | `True` | Keep workers alive |

### Checkpointing

| Argument | Default | Description |
|----------|---------|-------------|
| `--save-freq` | `10` | Save checkpoint every N epochs |
| `--no-resume` | `False` | Start fresh |
| `--skip-eval` | `False` | Skip evaluation |
| `--device` | `auto` | Device (cuda/cpu) |

## GPU Configuration

### Memory Usage Guide

| GPU | VRAM | Recommended Config | Epoch Time |
|-----|------|-------------------|------------|
| RTX 3060 | 12GB | `--batch-size 4 --filters 64 --amp` | ~25s |
| RTX 3070 | 8GB | `--batch-size 2 --filters 64 --amp` | ~30s |
| RTX 3080 | 10GB | `--batch-size 4 --filters 64 --amp` | ~20s |
| RTX 3090 | 24GB | `--batch-size 8 --filters 96 --amp` | ~15s |
| RTX 4090 | 24GB | `--batch-size 8 --filters 96 --amp` | ~10s |
| A100 | 40GB | `--batch-size 16 --filters 96 --amp` | ~8s |
| A100 | 80GB | `--batch-size 32 --filters 128 --amp` | ~6s |

### Model Size Reference

| Filters | Parameters | VRAM (batch=4, AMP) |
|---------|------------|---------------------|
| 32 | ~19M | ~4 GB |
| 64 | ~77M | ~8 GB |
| 96 | ~173M | ~11 GB |
| 128 | ~308M | ~16 GB |

## Evaluation Metrics

### Dose Score (MAE)

Mean Absolute Error between predicted and reference dose across all voxels in the possible dose mask.

```
Dose Score = Σ|predicted - reference| / num_voxels
```

- **Units**: Gray (Gy)
- **Lower is better**
- **Competition benchmark**: ~2-3 Gy

### DVH Score

Mean absolute difference in Dose-Volume Histogram metrics:

| Structure Type | Metrics |
|----------------|---------|
| Organs at Risk | D_0.1_cc (dose to hottest 0.1cc), Mean dose |
| Targets (PTVs) | D_99, D_95, D_1 |

- **Units**: Gray (Gy)
- **Lower is better**
- **Competition benchmark**: ~1.5-2.5 Gy

### Expected Results

| Training | Epochs | Dose Score | DVH Score |
|----------|--------|------------|-----------|
| Quick test (10% data) | 50 | ~15-20 Gy | ~20-25 Gy |
| Short training | 100 | ~12-15 Gy | ~15-20 Gy |
| Medium training | 300 | ~8-12 Gy | ~10-15 Gy |
| Long training | 500+ | ~5-8 Gy | ~6-10 Gy |
| Competition winners | - | ~2-3 Gy | ~1.5-2.5 Gy |

## Output Files

Results are saved to `results/{model_name}_{data_pct}pct_{timestamp}/`:

```
results/monai_unet_100pct_20260104_120000/
├── models/
│   ├── best_model.pt              # Best validation loss
│   ├── epoch_50.pt                # Periodic checkpoints
│   └── epoch_100.pt               # Final checkpoint
├── exports/
│   ├── training_summary.json      # Complete training summary
│   ├── evaluation_results.json    # Validation metrics
│   ├── test_results.json          # Test set metrics
│   └── losses.csv                 # Epoch-by-epoch losses
├── validation-predictions/        # Predicted dose CSVs
├── test-predictions/              # Test set predictions
└── training_history.png           # Loss curve plot
```

### Sample Training Summary

```json
{
  "timestamp": "2026-01-04T12:00:00",
  "model_name": "monai_unet_100pct_20260104_120000",
  "config": {
    "num_epochs": 400,
    "batch_size": 4,
    "learning_rate": 0.0003,
    "num_filters": 64,
    "scheduler_type": "onecycle",
    "use_amp": true,
    "gradient_accumulation_steps": 4,
    "effective_batch_size": 16
  },
  "model_summary": {
    "total_params": 77000000,
    "device": "cuda",
    "use_amp": true
  },
  "training": {
    "best_val_loss": 10.5,
    "best_epoch": 350,
    "total_time_hours": 2.8
  },
  "evaluation": {
    "dose_score": 8.2,
    "dvh_score": 12.1,
    "num_patients": 40
  }
}
```

## Legacy Code

The original TensorFlow/Keras implementation is preserved in `legacy/`:

```
legacy/
├── provided_code/         # Original Python modules
├── main.py                # Original training script
├── main_notebook.ipynb    # Original Jupyter notebook
└── requirements.txt       # TensorFlow dependencies
```

## Competition Results

The OpenKBP Challenge (Feb-June 2020) attracted 195 participants from 28 countries.

### Winners

**First Place (Dose & DVH)**: LSL AnHui University, China
[\[GitHub\]](https://github.com/LSL000UD/RTDosePrediction) [\[Paper\]](https://aapm.onlinelibrary.wiley.com/doi/full/10.1002/mp.15034)

**Runner-up (Dose)**: SuperPod, MD Anderson Cancer Center, USA
[\[Paper\]](https://aapm.onlinelibrary.wiley.com/doi/full/10.1002/mp.14827)

**Runner-up (DVH)**: PTV - Prediction Team Vienna, Austria
[\[Paper\]](https://aapm.onlinelibrary.wiley.com/doi/full/10.1002/mp.14774)

### Final Leaderboard

![](read-me-images/final_leaderboard.png)

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
--batch-size 2

# Enable gradient checkpointing
--grad-checkpoint

# Reduce model size
--filters 32
```

### Training Instability

```bash
# Lower learning rate
--lr 1e-4

# Increase gradient clipping
--grad-clip 0.5

# Use plateau scheduler for adaptive LR
--scheduler plateau --lr-patience 10
```

### Slow Data Loading

```bash
# Increase workers
--num-workers 8

# Enable persistent workers
--persistent-workers

# Increase prefetch
--prefetch 4
```

## License

See [LICENSE](LICENSE) file.

## Acknowledgments

- Original OpenKBP organizers: Aaron Babier, Binghao Zhang, Rafid Mahmood, Timothy Chan (University of Toronto); Andrea McNiven, Thomas Purdie (Princess Margaret Cancer Center); Kevin Moore (UC San Diego)
- Supported by [The American Association of Physicists in Medicine](https://www.aapm.org/GrandChallenge/OpenKBP/)