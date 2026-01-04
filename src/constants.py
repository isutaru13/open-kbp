"""
Project-wide constants for OpenKBP dose prediction.
"""

from typing import Dict, List, Tuple

# Patient volume shape (128 x 128 x 128 voxels)
PATIENT_SHAPE: Tuple[int, int, int] = (128, 128, 128)

# Regions of Interest (ROIs)
ROIS: Dict[str, List[str]] = {
    "oars": [
        "Brainstem",
        "SpinalCord",
        "RightParotid",
        "LeftParotid",
        "Esophagus",
        "Larynx",
        "Mandible",
    ],
    "targets": [
        "PTV56",
        "PTV63",
        "PTV70",
    ],
}

# Flattened list of all ROIs
FULL_ROI_LIST: List[str] = ROIS["oars"] + ROIS["targets"]

# Number of ROIs
NUM_ROIS: int = len(FULL_ROI_LIST)

# Number of input channels for the model (1 CT + NUM_ROIS structure masks)
NUM_INPUT_CHANNELS: int = 1 + NUM_ROIS

# Default training configuration
DEFAULT_CONFIG: Dict = {
    "model_name": "monai_unet",
    "num_epochs": 100,
    "batch_size": 2,
    "learning_rate": 1e-4,
    "num_filters": 32,
    "save_frequency": 10,
    "num_workers": 4,
    "resume": True,
}
