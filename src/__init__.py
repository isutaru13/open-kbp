"""
OpenKBP MONAI-based dose prediction package.

This package provides tools for training 3D U-Net models to predict
radiation dose distributions for head-and-neck cancer patients.
"""

from .constants import (
    DEFAULT_CONFIG,
    FULL_ROI_LIST,
    NUM_INPUT_CHANNELS,
    NUM_ROIS,
    PATIENT_SHAPE,
    ROIS,
)
from .dataset import OpenKBPDataset, get_patient_dirs
from .evaluation import (
    compute_dose_score,
    compute_dvh_score,
    evaluate_patient,
    evaluate_predictions,
    print_evaluation_results,
)
from .export import (
    append_to_experiment_log,
    create_results_directory,
    export_config,
    export_evaluation_results,
    export_losses_csv,
    export_patient_results,
    export_training_history,
    export_training_summary,
    load_evaluation_results,
    load_training_history,
)
from .losses import (
    CombinedLoss,
    MaskedHuberLoss,
    MaskedMAELoss,
    MaskedMSELoss,
    get_loss_function,
)
from .model import DosePredictionModel
from .hd_unet import HDUNet, HDUNetLite, get_hd_unet
from .hd_unet_model import HDUNetDosePredictionModel
from .transforms import (
    get_full_augment_transforms,
    get_inference_transforms,
    get_intensity_augment_transforms,
    get_no_augment_transforms,
    get_transforms,
)
from .visualization import (
    create_visualization_report,
    plot_ct_with_dose_overlay,
    plot_dose_comparison,
    plot_dose_slice,
    plot_dose_volume_histogram,
    plot_multi_slice_comparison,
    plot_training_history,
)

__all__ = [
    # Constants
    "PATIENT_SHAPE",
    "ROIS",
    "FULL_ROI_LIST",
    "NUM_ROIS",
    "NUM_INPUT_CHANNELS",
    "DEFAULT_CONFIG",
    # Dataset
    "OpenKBPDataset",
    "get_patient_dirs",
    # Transforms
    "get_transforms",
    "get_inference_transforms",
    "get_no_augment_transforms",
    "get_intensity_augment_transforms",
    "get_full_augment_transforms",
    # Losses
    "MaskedMAELoss",
    "MaskedMSELoss",
    "MaskedHuberLoss",
    "CombinedLoss",
    "get_loss_function",
    # Model
    "DosePredictionModel",
    # HD U-Net
    "HDUNet",
    "HDUNetLite",
    "get_hd_unet",
    "HDUNetDosePredictionModel",
    # Evaluation
    "compute_dose_score",
    "compute_dvh_score",
    "evaluate_patient",
    "evaluate_predictions",
    "print_evaluation_results",
    # Export
    "export_training_history",
    "export_evaluation_results",
    "export_config",
    "export_patient_results",
    "export_training_summary",
    "export_losses_csv",
    "load_training_history",
    "load_evaluation_results",
    "create_results_directory",
    "append_to_experiment_log",
    # Visualization
    "plot_training_history",
    "plot_dose_slice",
    "plot_dose_comparison",
    "plot_ct_with_dose_overlay",
    "plot_dose_volume_histogram",
    "plot_multi_slice_comparison",
    "create_visualization_report",
]
