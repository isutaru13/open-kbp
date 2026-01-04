"""
OpenKBP Dataset module for PyTorch/MONAI.

This module provides the Dataset class for loading OpenKBP patient data
in a format suitable for training deep learning models.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Constants
PATIENT_SHAPE = (128, 128, 128)
ROIS = {
    "oars": [
        "Brainstem",
        "SpinalCord",
        "RightParotid",
        "LeftParotid",
        "Esophagus",
        "Larynx",
        "Mandible",
    ],
    "targets": ["PTV56", "PTV63", "PTV70"],
}
FULL_ROI_LIST = ROIS["oars"] + ROIS["targets"]
NUM_ROIS = len(FULL_ROI_LIST)


def load_sparse_file(file_path: Path) -> Dict[str, np.ndarray]:
    """Load a sparse CSV file and return indices and data.

    Args:
        file_path: Path to the CSV file

    Returns:
        Dictionary with 'indices' and 'data' keys
    """
    df = pd.read_csv(file_path, index_col=0)
    if df.isnull().values.any():
        # Data is a mask (index only)
        return {"indices": np.array(df.index).squeeze(), "data": None}
    else:
        # Data is sparse (index + values)
        return {"indices": df.index.values, "data": df["data"].values}


def sparse_to_dense(
    sparse_data: Dict[str, np.ndarray],
    shape: Tuple[int, ...],
    default_value: float = 0.0,
) -> np.ndarray:
    """Convert sparse representation to dense array.

    Args:
        sparse_data: Dictionary with 'indices' and 'data' keys
        shape: Target shape for the dense array
        default_value: Value to fill non-sparse locations

    Returns:
        Dense numpy array of the specified shape
    """
    dense = np.full(np.prod(shape), default_value, dtype=np.float32)
    if sparse_data["data"] is not None:
        np.put(dense, sparse_data["indices"], sparse_data["data"])
    else:
        np.put(dense, sparse_data["indices"], 1.0)
    return dense.reshape(shape)


def get_patient_dirs(data_dir: Path) -> List[Path]:
    """Get all patient directories from a data directory.

    Args:
        data_dir: Path to directory containing patient folders

    Returns:
        Sorted list of patient directory paths
    """
    patient_dirs = sorted(
        [d for d in data_dir.iterdir() if d.is_dir() and d.stem.startswith("pt_")]
    )
    return patient_dirs


class OpenKBPDataset(Dataset):
    """PyTorch Dataset for OpenKBP data.

    This dataset loads CT images, structure masks, and dose distributions
    for head-and-neck cancer patients from the OpenKBP challenge.

    Attributes:
        patient_dirs: List of paths to patient directories
        transform: Optional transforms to apply to the data
        include_dose: Whether to load dose data (set False for inference)
    """

    def __init__(
        self,
        patient_dirs: List[Path],
        transform=None,
        include_dose: bool = True,
    ):
        """Initialize the dataset.

        Args:
            patient_dirs: List of paths to patient directories
            transform: Optional MONAI transforms to apply
            include_dose: Whether to load dose data (False for inference)
        """
        self.patient_dirs = patient_dirs
        self.transform = transform
        self.include_dose = include_dose

    def __len__(self) -> int:
        """Return the number of patients in the dataset."""
        return len(self.patient_dirs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load and return a single patient's data.

        Args:
            idx: Index of the patient to load

        Returns:
            Dictionary containing:
                - image: CT + structure masks (C, H, W, D)
                - mask: Possible dose mask (1, H, W, D)
                - patient_id: Patient identifier string
                - dose: (optional) Dose distribution (1, H, W, D)
        """
        patient_dir = self.patient_dirs[idx]
        patient_id = patient_dir.stem

        # Load CT
        ct_path = patient_dir / "ct.csv"
        ct_sparse = load_sparse_file(ct_path)
        ct = sparse_to_dense(ct_sparse, PATIENT_SHAPE)

        # Load structure masks
        structure_masks = np.zeros((*PATIENT_SHAPE, NUM_ROIS), dtype=np.float32)
        for roi_idx, roi_name in enumerate(FULL_ROI_LIST):
            roi_path = patient_dir / f"{roi_name}.csv"
            if roi_path.exists():
                roi_sparse = load_sparse_file(roi_path)
                roi_mask = sparse_to_dense(roi_sparse, PATIENT_SHAPE)
                structure_masks[..., roi_idx] = roi_mask

        # Load possible dose mask
        mask_path = patient_dir / "possible_dose_mask.csv"
        mask_sparse = load_sparse_file(mask_path)
        possible_dose_mask = sparse_to_dense(mask_sparse, PATIENT_SHAPE)

        # Prepare input: CT (1 channel) + structure masks (NUM_ROIS channels)
        # Shape: (C, H, W, D) where C = 1 + NUM_ROIS
        ct_expanded = ct[np.newaxis, ...]  # (1, 128, 128, 128)
        structure_masks_transposed = np.transpose(
            structure_masks, (3, 0, 1, 2)
        )  # (NUM_ROIS, 128, 128, 128)
        image = np.concatenate(
            [ct_expanded, structure_masks_transposed], axis=0
        )  # (1 + NUM_ROIS, 128, 128, 128)

        # Prepare mask
        mask = possible_dose_mask[np.newaxis, ...]  # (1, 128, 128, 128)

        data = {
            "image": image,
            "mask": mask,
            "patient_id": patient_id,
        }

        # Load dose if training
        if self.include_dose:
            dose_path = patient_dir / "dose.csv"
            dose_sparse = load_sparse_file(dose_path)
            dose = sparse_to_dense(dose_sparse, PATIENT_SHAPE)
            dose = dose[np.newaxis, ...]  # (1, 128, 128, 128)
            data["dose"] = dose

        # Apply transforms
        if self.transform:
            data = self.transform(data)

        return data
