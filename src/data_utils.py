"""
Data utilities for loading and processing OpenKBP sparse data.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def load_sparse_file(file_path: Path) -> Dict[str, Optional[np.ndarray]]:
    """
    Load a sparse CSV file and return indices and data.

    Args:
        file_path: Path to the CSV file

    Returns:
        Dictionary with 'indices' and 'data' keys.
        For mask files, 'data' will be None.
    """
    df = pd.read_csv(file_path, index_col=0)
    if df.isnull().values.any():
        # Data is a mask (index only)
        return {"indices": np.array(df.index).squeeze(), "data": None}
    else:
        # Data is sparse (index + values)
        return {"indices": df.index.values, "data": df["data"].values}


def sparse_to_dense(
    sparse_data: Dict[str, Optional[np.ndarray]],
    shape: Tuple[int, ...],
    default_value: float = 0.0,
) -> np.ndarray:
    """
    Convert sparse representation to dense array.

    Args:
        sparse_data: Dictionary with 'indices' and 'data' keys
        shape: Target shape for the dense array
        default_value: Value to fill non-indexed positions

    Returns:
        Dense numpy array of the specified shape
    """
    dense = np.full(np.prod(shape), default_value, dtype=np.float32)
    if sparse_data["data"] is not None:
        np.put(dense, sparse_data["indices"], sparse_data["data"])
    else:
        np.put(dense, sparse_data["indices"], 1.0)
    return dense.reshape(shape)


def dense_to_sparse(
    dense_array: np.ndarray,
    threshold: float = 0.0,
) -> Dict[str, np.ndarray]:
    """
    Convert dense array to sparse representation.

    Args:
        dense_array: Dense numpy array
        threshold: Values above this threshold are considered non-zero

    Returns:
        Dictionary with 'indices' and 'data' keys
    """
    flat_array = dense_array.flatten()
    nonzero_mask = flat_array > threshold
    indices = np.where(nonzero_mask)[0]
    values = flat_array[nonzero_mask]
    return {"indices": indices, "data": values}


def save_sparse_csv(
    sparse_data: Dict[str, np.ndarray],
    file_path: Path,
) -> None:
    """
    Save sparse data to CSV file in OpenKBP format.

    Args:
        sparse_data: Dictionary with 'indices' and 'data' keys
        file_path: Path to save the CSV file
    """
    df = pd.DataFrame({"data": sparse_data["data"]}, index=sparse_data["indices"])
    df.to_csv(file_path)
