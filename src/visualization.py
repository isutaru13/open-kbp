"""
Visualization utilities for OpenKBP dose prediction.

This module provides functions for visualizing:
- Training history (loss curves)
- Dose distributions (2D slices)
- Comparison between predicted and reference doses
- DVH curves
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.figure import Figure


def plot_training_history(
    train_losses: List[float],
    val_losses: List[float],
    save_path: Optional[Path] = None,
    title: str = "Training History",
    figsize: Tuple[int, int] = (10, 6),
) -> Figure:
    """
    Plot training and validation loss curves.

    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        save_path: Path to save the figure (optional)
        title: Title for the plot
        figsize: Figure size (width, height)

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, "b-", label="Training Loss", linewidth=2)
    ax.plot(epochs, val_losses, "r-", label="Validation Loss", linewidth=2)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss (MAE)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add min validation loss annotation
    if val_losses:
        min_val_loss = min(val_losses)
        min_epoch = val_losses.index(min_val_loss) + 1
        ax.axhline(y=min_val_loss, color="r", linestyle="--", alpha=0.5)
        ax.annotate(
            f"Min: {min_val_loss:.4f} (epoch {min_epoch})",
            xy=(min_epoch, min_val_loss),
            xytext=(min_epoch + len(epochs) * 0.1, min_val_loss * 1.1),
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="red", alpha=0.5),
        )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Training history saved to {save_path}")

    return fig


def plot_dose_slice(
    dose: np.ndarray,
    slice_idx: Optional[int] = None,
    axis: int = 2,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "jet",
    title: str = "Dose Distribution",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (8, 8),
) -> Figure:
    """
    Plot a single slice of a 3D dose distribution.

    Args:
        dose: 3D dose array (H, W, D)
        slice_idx: Index of the slice to plot (default: middle slice)
        axis: Axis along which to slice (0, 1, or 2)
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        cmap: Colormap name
        title: Title for the plot
        save_path: Path to save the figure (optional)
        figsize: Figure size (width, height)

    Returns:
        Matplotlib Figure object
    """
    # Get slice
    if slice_idx is None:
        slice_idx = dose.shape[axis] // 2

    if axis == 0:
        slice_data = dose[slice_idx, :, :]
    elif axis == 1:
        slice_data = dose[:, slice_idx, :]
    else:
        slice_data = dose[:, :, slice_idx]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot dose
    im = ax.imshow(
        slice_data.T,
        cmap=cmap,
        vmin=vmin or 0,
        vmax=vmax or slice_data.max(),
        origin="lower",
        aspect="equal",
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label="Dose (Gy)")
    cbar.ax.tick_params(labelsize=10)

    ax.set_title(f"{title} (slice {slice_idx})", fontsize=14)
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_dose_comparison(
    pred_dose: np.ndarray,
    ref_dose: np.ndarray,
    mask: Optional[np.ndarray] = None,
    slice_idx: Optional[int] = None,
    axis: int = 2,
    vmax: Optional[float] = None,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (16, 5),
) -> Figure:
    """
    Plot comparison between predicted and reference dose distributions.

    Shows three panels: predicted dose, reference dose, and difference.

    Args:
        pred_dose: Predicted 3D dose array
        ref_dose: Reference 3D dose array
        mask: Optional mask for valid dose regions
        slice_idx: Index of the slice to plot (default: middle slice)
        axis: Axis along which to slice
        vmax: Maximum value for dose colormap
        save_path: Path to save the figure (optional)
        figsize: Figure size (width, height)

    Returns:
        Matplotlib Figure object
    """
    # Get slice index
    if slice_idx is None:
        slice_idx = pred_dose.shape[axis] // 2

    # Extract slices
    if axis == 0:
        pred_slice = pred_dose[slice_idx, :, :]
        ref_slice = ref_dose[slice_idx, :, :]
        mask_slice = mask[slice_idx, :, :] if mask is not None else None
    elif axis == 1:
        pred_slice = pred_dose[:, slice_idx, :]
        ref_slice = ref_dose[:, slice_idx, :]
        mask_slice = mask[:, slice_idx, :] if mask is not None else None
    else:
        pred_slice = pred_dose[:, :, slice_idx]
        ref_slice = ref_dose[:, :, slice_idx]
        mask_slice = mask[:, :, slice_idx] if mask is not None else None

    # Calculate difference
    diff_slice = pred_slice - ref_slice
    if mask_slice is not None:
        diff_slice = diff_slice * mask_slice

    # Determine color limits
    if vmax is None:
        vmax = max(pred_slice.max(), ref_slice.max())
    diff_max = max(abs(diff_slice.min()), abs(diff_slice.max()))

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Plot predicted dose
    im0 = axes[0].imshow(
        pred_slice.T, cmap="jet", vmin=0, vmax=vmax, origin="lower", aspect="equal"
    )
    axes[0].set_title("Predicted Dose", fontsize=12)
    plt.colorbar(im0, ax=axes[0], label="Dose (Gy)")

    # Plot reference dose
    im1 = axes[1].imshow(
        ref_slice.T, cmap="jet", vmin=0, vmax=vmax, origin="lower", aspect="equal"
    )
    axes[1].set_title("Reference Dose", fontsize=12)
    plt.colorbar(im1, ax=axes[1], label="Dose (Gy)")

    # Plot difference
    im2 = axes[2].imshow(
        diff_slice.T,
        cmap="RdBu_r",
        vmin=-diff_max,
        vmax=diff_max,
        origin="lower",
        aspect="equal",
    )
    axes[2].set_title("Difference (Pred - Ref)", fontsize=12)
    plt.colorbar(im2, ax=axes[2], label="Dose Difference (Gy)")

    # Calculate statistics
    mae = (
        np.mean(np.abs(diff_slice[mask_slice > 0]))
        if mask_slice is not None
        else np.mean(np.abs(diff_slice))
    )
    fig.suptitle(
        f"Dose Comparison (slice {slice_idx}) - MAE: {mae:.3f} Gy", fontsize=14
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_ct_with_dose_overlay(
    ct: np.ndarray,
    dose: np.ndarray,
    slice_idx: Optional[int] = None,
    axis: int = 2,
    dose_threshold: float = 0.1,
    dose_alpha: float = 0.5,
    ct_window: Tuple[float, float] = (-200, 400),
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 10),
) -> Figure:
    """
    Plot CT image with dose overlay.

    Args:
        ct: 3D CT array
        dose: 3D dose array
        slice_idx: Index of the slice to plot (default: middle slice)
        axis: Axis along which to slice
        dose_threshold: Minimum dose value to display
        dose_alpha: Transparency of dose overlay
        ct_window: CT window (min, max) in HU
        save_path: Path to save the figure (optional)
        figsize: Figure size (width, height)

    Returns:
        Matplotlib Figure object
    """
    # Get slice index
    if slice_idx is None:
        slice_idx = ct.shape[axis] // 2

    # Extract slices
    if axis == 0:
        ct_slice = ct[slice_idx, :, :]
        dose_slice = dose[slice_idx, :, :]
    elif axis == 1:
        ct_slice = ct[:, slice_idx, :]
        dose_slice = dose[:, slice_idx, :]
    else:
        ct_slice = ct[:, :, slice_idx]
        dose_slice = dose[:, :, slice_idx]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot CT
    ax.imshow(
        ct_slice.T,
        cmap="gray",
        vmin=ct_window[0],
        vmax=ct_window[1],
        origin="lower",
        aspect="equal",
    )

    # Create masked dose for overlay
    dose_masked = np.ma.masked_where(dose_slice < dose_threshold, dose_slice)

    # Overlay dose
    im = ax.imshow(
        dose_masked.T,
        cmap="jet",
        alpha=dose_alpha,
        vmin=0,
        vmax=dose_slice.max(),
        origin="lower",
        aspect="equal",
    )

    plt.colorbar(im, ax=ax, label="Dose (Gy)")
    ax.set_title(f"CT with Dose Overlay (slice {slice_idx})", fontsize=14)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_dose_volume_histogram(
    dose: np.ndarray,
    roi_masks: Dict[str, np.ndarray],
    num_bins: int = 100,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> Figure:
    """
    Plot Dose-Volume Histogram (DVH) for multiple ROIs.

    Args:
        dose: 3D dose array
        roi_masks: Dictionary mapping ROI names to binary masks
        num_bins: Number of bins for histogram
        save_path: Path to save the figure (optional)
        figsize: Figure size (width, height)

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    max_dose = dose.max()
    dose_bins = np.linspace(0, max_dose, num_bins)

    colors = plt.cm.tab10(np.linspace(0, 1, len(roi_masks)))

    for (roi_name, roi_mask), color in zip(roi_masks.items(), colors):
        if roi_mask.sum() == 0:
            continue

        roi_dose = dose[roi_mask.astype(bool)]
        total_volume = len(roi_dose)

        # Calculate cumulative histogram (DVH)
        volumes = []
        for dose_threshold in dose_bins:
            volume_above = np.sum(roi_dose >= dose_threshold) / total_volume * 100
            volumes.append(volume_above)

        ax.plot(dose_bins, volumes, label=roi_name, color=color, linewidth=2)

    ax.set_xlabel("Dose (Gy)", fontsize=12)
    ax.set_ylabel("Volume (%)", fontsize=12)
    ax.set_title("Dose-Volume Histogram", fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_dose)
    ax.set_ylim(0, 100)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_multi_slice_comparison(
    pred_dose: np.ndarray,
    ref_dose: np.ndarray,
    num_slices: int = 5,
    axis: int = 2,
    save_path: Optional[Path] = None,
    figsize: Optional[Tuple[int, int]] = None,
) -> Figure:
    """
    Plot multiple slices comparing predicted and reference doses.

    Args:
        pred_dose: Predicted 3D dose array
        ref_dose: Reference 3D dose array
        num_slices: Number of slices to display
        axis: Axis along which to slice
        save_path: Path to save the figure (optional)
        figsize: Figure size (optional, auto-calculated if None)

    Returns:
        Matplotlib Figure object
    """
    # Calculate slice indices (evenly spaced, excluding edges)
    total_slices = pred_dose.shape[axis]
    slice_indices = np.linspace(
        total_slices * 0.2, total_slices * 0.8, num_slices, dtype=int
    )

    # Auto-calculate figure size
    if figsize is None:
        figsize = (4 * num_slices, 8)

    fig, axes = plt.subplots(2, num_slices, figsize=figsize)

    vmax = max(pred_dose.max(), ref_dose.max())

    for i, slice_idx in enumerate(slice_indices):
        # Extract slices
        if axis == 0:
            pred_slice = pred_dose[slice_idx, :, :]
            ref_slice = ref_dose[slice_idx, :, :]
        elif axis == 1:
            pred_slice = pred_dose[:, slice_idx, :]
            ref_slice = ref_dose[:, slice_idx, :]
        else:
            pred_slice = pred_dose[:, :, slice_idx]
            ref_slice = ref_dose[:, :, slice_idx]

        # Plot predicted
        axes[0, i].imshow(
            pred_slice.T, cmap="jet", vmin=0, vmax=vmax, origin="lower", aspect="equal"
        )
        axes[0, i].set_title(f"Pred (z={slice_idx})", fontsize=10)
        axes[0, i].axis("off")

        # Plot reference
        axes[1, i].imshow(
            ref_slice.T, cmap="jet", vmin=0, vmax=vmax, origin="lower", aspect="equal"
        )
        axes[1, i].set_title(f"Ref (z={slice_idx})", fontsize=10)
        axes[1, i].axis("off")

    # Add row labels
    axes[0, 0].set_ylabel("Predicted", fontsize=12)
    axes[1, 0].set_ylabel("Reference", fontsize=12)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def create_visualization_report(
    pred_dose: np.ndarray,
    ref_dose: np.ndarray,
    ct: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    roi_masks: Optional[Dict[str, np.ndarray]] = None,
    output_dir: Path = Path("visualizations"),
    patient_id: str = "patient",
) -> None:
    """
    Create a complete visualization report for a patient.

    Args:
        pred_dose: Predicted 3D dose array
        ref_dose: Reference 3D dose array
        ct: Optional CT array
        mask: Optional possible dose mask
        roi_masks: Optional dictionary of ROI masks
        output_dir: Directory to save visualizations
        patient_id: Patient identifier for file naming
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find slice with maximum dose
    max_dose_idx = np.unravel_index(np.argmax(ref_dose), ref_dose.shape)[2]

    # 1. Dose comparison at max dose slice
    plot_dose_comparison(
        pred_dose,
        ref_dose,
        mask=mask,
        slice_idx=max_dose_idx,
        save_path=output_dir / f"{patient_id}_dose_comparison.png",
    )
    plt.close()

    # 2. Multi-slice comparison
    plot_multi_slice_comparison(
        pred_dose,
        ref_dose,
        num_slices=5,
        save_path=output_dir / f"{patient_id}_multi_slice.png",
    )
    plt.close()

    # 3. CT with dose overlay (if CT available)
    if ct is not None:
        plot_ct_with_dose_overlay(
            ct,
            pred_dose,
            slice_idx=max_dose_idx,
            save_path=output_dir / f"{patient_id}_ct_overlay.png",
        )
        plt.close()

    # 4. DVH (if ROI masks available)
    if roi_masks:
        plot_dose_volume_histogram(
            pred_dose,
            roi_masks,
            save_path=output_dir / f"{patient_id}_dvh.png",
        )
        plt.close()

    print(f"Visualization report saved to {output_dir}")
