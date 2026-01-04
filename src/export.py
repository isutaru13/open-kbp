"""
Results export module for OpenKBP dose prediction.

This module provides functions for exporting training results,
evaluation metrics, and model configurations to JSON and CSV formats.
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def export_training_history(
    train_losses: List[float],
    val_losses: List[float],
    output_path: Path,
    additional_info: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Export training history to JSON file.

    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        output_path: Path to save the JSON file
        additional_info: Optional additional information to include
    """
    history = {
        "timestamp": datetime.now().isoformat(),
        "num_epochs": len(train_losses),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": min(val_losses) if val_losses else None,
        "best_epoch": val_losses.index(min(val_losses)) + 1 if val_losses else None,
        "final_train_loss": train_losses[-1] if train_losses else None,
        "final_val_loss": val_losses[-1] if val_losses else None,
    }

    if additional_info:
        history.update(additional_info)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"Training history exported to {output_path}")


def export_evaluation_results(
    results: Dict[str, Any],
    output_path: Path,
    model_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Export evaluation results to JSON file.

    Args:
        results: Dictionary containing evaluation metrics
        output_path: Path to save the JSON file
        model_name: Optional model name to include
        config: Optional model configuration to include
    """
    export_data = {
        "timestamp": datetime.now().isoformat(),
        "model_name": model_name,
        "results": results,
    }

    if config:
        export_data["config"] = config

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(export_data, f, indent=2)

    print(f"Evaluation results exported to {output_path}")


def export_config(
    config: Dict[str, Any],
    output_path: Path,
) -> None:
    """
    Export model configuration to JSON file.

    Args:
        config: Dictionary containing model configuration
        output_path: Path to save the JSON file
    """
    export_data = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(export_data, f, indent=2)

    print(f"Configuration exported to {output_path}")


def export_patient_results(
    patient_results: List[Dict[str, Any]],
    output_path: Path,
    format: str = "json",
) -> None:
    """
    Export per-patient evaluation results.

    Args:
        patient_results: List of dictionaries with patient-level results
        output_path: Path to save the file
        format: Output format ('json' or 'csv')
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format.lower() == "json":
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "num_patients": len(patient_results),
            "patients": patient_results,
        }
        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2)

    elif format.lower() == "csv":
        if not patient_results:
            print("No patient results to export")
            return

        # Get all unique keys from results
        all_keys = set()
        for result in patient_results:
            all_keys.update(result.keys())
        fieldnames = sorted(all_keys)

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(patient_results)

    else:
        raise ValueError(f"Unsupported format: {format}. Use 'json' or 'csv'.")

    print(f"Patient results exported to {output_path}")


def export_training_summary(
    model_name: str,
    config: Dict[str, Any],
    train_losses: List[float],
    val_losses: List[float],
    evaluation_results: Dict[str, Any],
    output_path: Path,
    model_summary: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Export complete training summary to JSON file.

    This creates a comprehensive summary including configuration,
    training history, and evaluation results.

    Args:
        model_name: Name of the model
        config: Model configuration dictionary
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        evaluation_results: Dictionary with evaluation metrics
        output_path: Path to save the JSON file
        model_summary: Optional model architecture summary
    """
    summary = {
        "timestamp": datetime.now().isoformat(),
        "model_name": model_name,
        "config": config,
        "model_summary": model_summary,
        "training": {
            "num_epochs": len(train_losses),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": min(val_losses) if val_losses else None,
            "best_epoch": val_losses.index(min(val_losses)) + 1 if val_losses else None,
            "final_train_loss": train_losses[-1] if train_losses else None,
            "final_val_loss": val_losses[-1] if val_losses else None,
        },
        "evaluation": evaluation_results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Training summary exported to {output_path}")


def load_training_history(input_path: Path) -> Dict[str, Any]:
    """
    Load training history from JSON file.

    Args:
        input_path: Path to the JSON file

    Returns:
        Dictionary containing training history
    """
    with open(input_path, "r") as f:
        return json.load(f)


def load_evaluation_results(input_path: Path) -> Dict[str, Any]:
    """
    Load evaluation results from JSON file.

    Args:
        input_path: Path to the JSON file

    Returns:
        Dictionary containing evaluation results
    """
    with open(input_path, "r") as f:
        return json.load(f)


def create_results_directory(
    base_dir: Path,
    model_name: str,
    create_subdirs: bool = True,
) -> Dict[str, Path]:
    """
    Create a structured results directory for a model.

    Args:
        base_dir: Base results directory
        model_name: Name of the model
        create_subdirs: Whether to create subdirectories

    Returns:
        Dictionary mapping directory names to paths
    """
    model_dir = base_dir / model_name

    dirs = {
        "root": model_dir,
        "models": model_dir / "models",
        "predictions": model_dir / "predictions",
        "visualizations": model_dir / "visualizations",
        "exports": model_dir / "exports",
    }

    if create_subdirs:
        for dir_path in dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

    return dirs


def export_losses_csv(
    train_losses: List[float],
    val_losses: List[float],
    output_path: Path,
) -> None:
    """
    Export training and validation losses to CSV file.

    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        output_path: Path to save the CSV file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for epoch, (train_loss, val_loss) in enumerate(
            zip(train_losses, val_losses), start=1
        ):
            writer.writerow([epoch, train_loss, val_loss])

    print(f"Losses exported to {output_path}")


def append_to_experiment_log(
    log_path: Path,
    model_name: str,
    config: Dict[str, Any],
    results: Dict[str, Any],
) -> None:
    """
    Append experiment results to a running log file.

    Useful for tracking multiple experiments over time.

    Args:
        log_path: Path to the experiment log file
        model_name: Name of the model
        config: Model configuration
        results: Evaluation results
    """
    # Load existing log or create new
    if log_path.exists():
        with open(log_path, "r") as f:
            log = json.load(f)
    else:
        log = {"experiments": []}

    # Add new experiment
    experiment = {
        "timestamp": datetime.now().isoformat(),
        "model_name": model_name,
        "config": config,
        "results": results,
    }
    log["experiments"].append(experiment)

    # Save updated log
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"Experiment logged to {log_path}")
