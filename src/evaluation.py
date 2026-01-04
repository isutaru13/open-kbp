"""
Evaluation module for computing OpenKBP dose prediction metrics.

This module provides functions to evaluate dose predictions against
reference doses using the metrics from the OpenKBP challenge.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from .constants import FULL_ROI_LIST, PATIENT_SHAPE, ROIS
from .data_utils import load_sparse_file, sparse_to_dense


def compute_dose_score(
    pred_dose: np.ndarray,
    ref_dose: np.ndarray,
    mask: np.ndarray,
) -> float:
    """
    Compute the dose score (mean absolute error) between predicted and reference dose.

    Args:
        pred_dose: Predicted dose array
        ref_dose: Reference dose array
        mask: Possible dose mask

    Returns:
        Dose score (MAE)
    """
    dose_error = np.sum(np.abs(ref_dose - pred_dose)) / (np.sum(mask) + 1e-8)
    return float(dose_error)


def compute_dvh_metric(
    dose: np.ndarray,
    roi_mask: np.ndarray,
    metric: str,
    voxel_size: float = 1.0,
) -> float:
    """
    Compute a single DVH metric for a region of interest.

    Args:
        dose: 3D dose array
        roi_mask: Binary mask for the ROI
        metric: Metric to compute ('D_0.1_cc', 'mean', 'D_99', 'D_95', 'D_1')
        voxel_size: Volume of a single voxel in cc

    Returns:
        DVH metric value
    """
    roi_dose = dose[roi_mask.astype(bool)]

    if len(roi_dose) == 0:
        return np.nan

    if metric == "D_0.1_cc":
        # Dose to the hottest 0.1 cc
        voxels_in_tenth_cc = max(1, int(np.round(0.1 / voxel_size)))
        fractional_volume = 100 - (voxels_in_tenth_cc / len(roi_dose) * 100)
        return float(np.percentile(roi_dose, max(0, fractional_volume)))
    elif metric == "mean":
        return float(np.mean(roi_dose))
    elif metric == "D_99":
        # Dose covering 99% of the volume
        return float(np.percentile(roi_dose, 1))
    elif metric == "D_95":
        # Dose covering 95% of the volume
        return float(np.percentile(roi_dose, 5))
    elif metric == "D_1":
        # Dose covering 1% of the volume (near-max dose)
        return float(np.percentile(roi_dose, 99))
    else:
        raise ValueError(f"Unknown metric: {metric}")


def get_dvh_metrics_for_roi(roi_name: str) -> List[str]:
    """
    Get the list of DVH metrics to compute for a given ROI.

    Args:
        roi_name: Name of the ROI

    Returns:
        List of metric names
    """
    if roi_name in ROIS["oars"]:
        return ["D_0.1_cc", "mean"]
    elif roi_name in ROIS["targets"]:
        return ["D_99", "D_95", "D_1"]
    else:
        raise ValueError(f"Unknown ROI: {roi_name}")


def compute_all_dvh_metrics(
    dose: np.ndarray,
    patient_dir: Path,
    voxel_size: float = 1.0,
) -> Dict[Tuple[str, str], float]:
    """
    Compute all DVH metrics for a patient.

    Args:
        dose: 3D dose array
        patient_dir: Path to patient directory containing ROI masks
        voxel_size: Volume of a single voxel in cc

    Returns:
        Dictionary mapping (metric, roi) tuples to values
    """
    metrics = {}

    for roi_name in FULL_ROI_LIST:
        roi_path = patient_dir / f"{roi_name}.csv"
        if not roi_path.exists():
            continue

        # Load ROI mask
        roi_sparse = load_sparse_file(roi_path)
        roi_mask = sparse_to_dense(roi_sparse, PATIENT_SHAPE)

        # Compute metrics for this ROI
        for metric in get_dvh_metrics_for_roi(roi_name):
            value = compute_dvh_metric(dose, roi_mask, metric, voxel_size)
            metrics[(metric, roi_name)] = value

    return metrics


def compute_dvh_score(
    pred_metrics: Dict[Tuple[str, str], float],
    ref_metrics: Dict[Tuple[str, str], float],
) -> float:
    """
    Compute the DVH score as mean absolute difference between metrics.

    Args:
        pred_metrics: Predicted DVH metrics
        ref_metrics: Reference DVH metrics

    Returns:
        DVH score (mean absolute difference)
    """
    differences = []
    for key in ref_metrics:
        if key in pred_metrics:
            ref_val = ref_metrics[key]
            pred_val = pred_metrics[key]
            if not np.isnan(ref_val) and not np.isnan(pred_val):
                differences.append(abs(ref_val - pred_val))

    if not differences:
        return np.nan

    return float(np.mean(differences))


def evaluate_patient(
    pred_path: Path,
    ref_dir: Path,
    patient_id: str,
) -> Dict[str, float]:
    """
    Evaluate predictions for a single patient.

    Args:
        pred_path: Path to predicted dose CSV file
        ref_dir: Path to reference data directory
        patient_id: Patient identifier

    Returns:
        Dictionary with 'dose_score' and 'dvh_score'
    """
    patient_dir = ref_dir / patient_id

    # Load predicted dose
    pred_sparse = load_sparse_file(pred_path)
    pred_dose = sparse_to_dense(pred_sparse, PATIENT_SHAPE)

    # Load reference dose
    ref_sparse = load_sparse_file(patient_dir / "dose.csv")
    ref_dose = sparse_to_dense(ref_sparse, PATIENT_SHAPE)

    # Load mask
    mask_sparse = load_sparse_file(patient_dir / "possible_dose_mask.csv")
    mask = sparse_to_dense(mask_sparse, PATIENT_SHAPE)

    # Load voxel dimensions to compute voxel volume
    voxel_dims_path = patient_dir / "voxel_dimensions.csv"
    if voxel_dims_path.exists():
        voxel_dims = np.loadtxt(voxel_dims_path)
        voxel_size = np.prod(voxel_dims) / 1000.0  # Convert mm^3 to cc
    else:
        voxel_size = 1.0

    # Compute dose score
    dose_score = compute_dose_score(pred_dose, ref_dose, mask)

    # Compute DVH metrics
    pred_dvh = compute_all_dvh_metrics(pred_dose, patient_dir, voxel_size)
    ref_dvh = compute_all_dvh_metrics(ref_dose, patient_dir, voxel_size)
    dvh_score = compute_dvh_score(pred_dvh, ref_dvh)

    return {
        "dose_score": dose_score,
        "dvh_score": dvh_score,
    }


def evaluate_predictions(
    pred_dir: Path,
    ref_dir: Path,
    patient_ids: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Evaluate predictions for multiple patients.

    Args:
        pred_dir: Directory containing prediction CSV files
        ref_dir: Directory containing reference patient data
        patient_ids: List of patient IDs to evaluate (None = all)
        verbose: Whether to show progress bar

    Returns:
        Dictionary with mean 'dose_score' and 'dvh_score'
    """
    # Get patient IDs if not provided
    if patient_ids is None:
        patient_ids = [p.stem for p in pred_dir.glob("pt_*.csv")]

    dose_scores = []
    dvh_scores = []

    iterator = tqdm(patient_ids, desc="Evaluating") if verbose else patient_ids
    for patient_id in iterator:
        pred_path = pred_dir / f"{patient_id}.csv"
        if not pred_path.exists():
            continue

        try:
            results = evaluate_patient(pred_path, ref_dir, patient_id)
            dose_scores.append(results["dose_score"])
            if not np.isnan(results["dvh_score"]):
                dvh_scores.append(results["dvh_score"])
        except Exception as e:
            if verbose:
                print(f"Error evaluating {patient_id}: {e}")
            continue

    return {
        "dose_score": float(np.mean(dose_scores)) if dose_scores else np.nan,
        "dvh_score": float(np.mean(dvh_scores)) if dvh_scores else np.nan,
        "num_patients": len(dose_scores),
    }


def print_evaluation_results(results: Dict[str, float]) -> None:
    """
    Print evaluation results in a formatted way.

    Args:
        results: Dictionary with evaluation metrics
    """
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Number of patients evaluated: {results.get('num_patients', 'N/A')}")
    print(f"Dose Score (MAE): {results['dose_score']:.4f}")
    print(f"DVH Score: {results['dvh_score']:.4f}")
    print("=" * 50)
