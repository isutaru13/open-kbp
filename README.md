# OpenKBP Dose Prediction with MONAI

![](read-me-images/aapm.png)

A MONAI-based 3D U-Net implementation for predicting radiation dose distributions in head-and-neck cancer patients. This is a refactored version of the [OpenKBP Grand Challenge](https://aapm.onlinelibrary.wiley.com/doi/epdf/10.1002/mp.14845) codebase, using PyTorch and MONAI instead of TensorFlow/Keras.

![](read-me-images/pipeline.png)

## Citation

If you use this dataset or code, please cite the original paper:

> A. Babier, B. Zhang, R. Mahmood, K.L. Moore, T.G. Purdie, A.L. McNiven, T.C.Y. Chan, "[OpenKBP: The open-access knowledge-based planning grand challenge and dataset](https://aapm.onlinelibrary.wiley.com/doi/epdf/10.1002/mp.14845)," _Medical Physics_, Vol. 48, pp. 5549-5561, 2021.

## Table of Contents

- [Features](#features)
- [Data](#data)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Quick Test](#quick-test)
  - [Full Training](#full-training)
  - [Command Line Options](#command-line-options)
- [GPU Configuration](#gpu-configuration)
- [Evaluation Metrics](#evaluation-metrics)
- [Output Files](#output-files)
- [Legacy Code](#legacy-code)
- [Competition Results](#competition-results)

## Features

- **MONAI 3D U-Net**: Modern PyTorch-based architecture with residual units
- **Modular Design**: Clean separation of dataset, model, transforms, losses, and evaluation
- **Flexible Training**: Configurable batch size, filters, learning rate, and epochs
- **Data Augmentation**: Random flips, rotations, and Gaussian noise
- **Automatic Checkpointing**: Saves best model and periodic checkpoints
- **Comprehensive Evaluation**: Dose score (MAE) and DVH metrics
- **Result Exports**: JSON summaries, CSV losses, and visualization plots
- **Timing & ETA**: Real-time epoch timing with estimated completion time

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
│   ├── model.py                  # DosePredictionModel trainer
│   ├── transforms.py             # MONAI data transforms
│   └── visualization.py          # Training plots and dose visualization
├── provided-data/                # Patient data
│   ├── train-pats/               # 200 training patients
│   ├── validation-pats/          # 40 validation patients
│   └── test-pats/                # 100 test patients
├── legacy/                       # Original TensorFlow/Keras code
├── results/                      # Training outputs (gitignored)
├── train_monai.py                # Main training script
├── requirements.txt              # Python dependencies
└── README.md
```

## Installation

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA (recommended)

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

## Usage

### Quick Test

Test the setup with 10% of data:

```bash
python train_monai.py --data-fraction 0.1 --epochs 5
```

### Full Training

```bash
# Default settings (100 epochs)
python train_monai.py --epochs 100

# Recommended for RTX 3060 12GB
python train_monai.py --epochs 100 --batch-size 4 --filters 64

# Recommended for A100 40GB
python train_monai.py --epochs 45 --batch-size 16 --filters 96 --num-workers 8
```

### Command Line Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-name` | `monai_unet` | Name for the model |
| `--epochs` | `100` | Number of training epochs |
| `--batch-size` | `2` | Batch size |
| `--lr` | `1e-4` | Learning rate |
| `--filters` | `32` | Initial U-Net filters (doubles each level) |
| `--save-freq` | `10` | Save checkpoint every N epochs |
| `--num-workers` | `4` | Data loading workers |
| `--data-fraction` | `1.0` | Fraction of data to use (for testing) |
| `--test-split` | `0.1` | Fraction of training data for test holdout |
| `--no-resume` | `False` | Start fresh (don't resume from checkpoint) |
| `--skip-eval` | `False` | Skip evaluation after training |
| `--device` | `auto` | Device (cuda/cpu) |

## GPU Configuration

### Memory Usage Guide

| GPU | VRAM | Recommended Config |
|-----|------|-------------------|
| RTX 3060 | 12GB | `--batch-size 4 --filters 64` (~77M params) |
| RTX 3090 | 24GB | `--batch-size 8 --filters 96` (~173M params) |
| A100 | 40GB | `--batch-size 16 --filters 96` (~173M params) |
| A100 | 80GB | `--batch-size 32 --filters 128` (~308M params) |

### Model Size Reference

| Filters | Parameters |
|---------|------------|
| 32 | ~19M |
| 64 | ~77M |
| 96 | ~173M |
| 128 | ~308M |

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

## Output Files

Results are saved to `results/{model_name}_{data_pct}pct_{timestamp}/`:

```
results/monai_unet_100pct_20260104_120000/
├── models/
│   ├── best_model.pt              # Best validation loss
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

### Sample JSON Output

```json
{
  "timestamp": "2026-01-04T12:00:00",
  "model_name": "monai_unet_100pct_20260104_120000",
  "config": {
    "num_epochs": 100,
    "batch_size": 4,
    "learning_rate": 0.0001,
    "num_filters": 64
  },
  "training": {
    "best_val_loss": 3.45,
    "best_epoch": 85,
    "total_time_minutes": 92.5
  },
  "evaluation": {
    "dose_score": 3.21,
    "dvh_score": 2.15,
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

## License

See [LICENSE](LICENSE) file.

## Acknowledgments

- Original OpenKBP organizers: Aaron Babier, Binghao Zhang, Rafid Mahmood, Timothy Chan (University of Toronto); Andrea McNiven, Thomas Purdie (Princess Margaret Cancer Center); Kevin Moore (UC San Diego)
- Supported by [The American Association of Physicists in Medicine](https://www.aapm.org/GrandChallenge/OpenKBP/)