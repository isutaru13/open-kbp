#!/usr/bin/env python3
"""
MONAI-based 3D U-Net training script for OpenKBP dose prediction.

This script trains a 3D U-Net model to predict radiation dose distributions
from CT images and structure masks for head-and-neck cancer patients.

Usage:
    python train_monai.py [--epochs N] [--batch-size N] [--lr LR] [--filters N]

    # Quick test with 10% of data
    python train_monai.py --data-fraction 0.1 --epochs 5
"""

import argparse
from datetime import datetime
from pathlib import Path

from torch.utils.data import DataLoader

from src import (
    DEFAULT_CONFIG,
    DosePredictionModel,
    OpenKBPDataset,
    get_inference_transforms,
    get_patient_dirs,
    get_transforms,
)
from src.evaluation import evaluate_predictions, print_evaluation_results
from src.export import (
    export_evaluation_results,
    export_losses_csv,
    export_training_summary,
)
from src.visualization import plot_training_history


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train MONAI 3D U-Net for OpenKBP dose prediction"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_CONFIG["model_name"],
        help="Name for the model (used for saving)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_CONFIG["num_epochs"],
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_CONFIG["batch_size"],
        help="Batch size for training",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_CONFIG["learning_rate"],
        help="Learning rate",
    )
    parser.add_argument(
        "--filters",
        type=int,
        default=DEFAULT_CONFIG["num_filters"],
        help="Initial number of filters in U-Net",
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=DEFAULT_CONFIG["save_frequency"],
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_CONFIG["num_workers"],
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start training from scratch (don't resume from checkpoint)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, default: auto)",
    )
    parser.add_argument(
        "--data-fraction",
        type=float,
        default=1.0,
        help="Fraction of training data to use (0.0-1.0, default: 1.0)",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.1,
        help="Fraction of training data to hold out for testing (default: 0.1)",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation after training",
    )
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Generate timestamp and run name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_pct = int(args.data_fraction * 100)
    run_name = f"{args.model_name}_{data_pct}pct_{timestamp}"

    # Build config dict for export
    config = {
        "model_name": args.model_name,
        "run_name": run_name,
        "timestamp": timestamp,
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "num_filters": args.filters,
        "save_frequency": args.save_freq,
        "num_workers": args.num_workers,
        "data_fraction": args.data_fraction,
        "test_split": args.test_split,
    }

    print("=" * 60)
    print("OpenKBP Dose Prediction with MONAI 3D U-Net")
    print("=" * 60)
    print(f"Run name: {run_name}")
    print(f"Model name: {args.model_name}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Initial filters: {args.filters}")
    print(f"Save frequency: {args.save_freq}")
    print(f"Num workers: {args.num_workers}")
    print(f"Data fraction: {args.data_fraction * 100:.0f}%")
    print(f"Test split: {args.test_split * 100:.0f}%")
    print(f"Resume: {not args.no_resume}")

    # Paths
    project_dir = Path(__file__).parent
    data_dir = project_dir / "provided-data"
    train_dir = data_dir / "train-pats"
    val_dir = data_dir / "validation-pats"
    results_dir = project_dir / "results"

    # Get patient directories
    train_patients = get_patient_dirs(train_dir)
    val_patients = get_patient_dirs(val_dir)

    # Apply data fraction for quick testing
    if args.data_fraction < 1.0:
        num_train = max(1, int(len(train_patients) * args.data_fraction))
        num_val = max(1, int(len(val_patients) * args.data_fraction))
        train_patients = train_patients[:num_train]
        val_patients = val_patients[:num_val]
        print(f"\n[Using {args.data_fraction * 100:.0f}% of data for quick testing]")

    # Hold out test set from training data
    test_patients = []
    if args.test_split > 0:
        num_test = max(1, int(len(train_patients) * args.test_split))
        test_patients = train_patients[-num_test:]  # Take from end
        train_patients = train_patients[:-num_test]  # Keep the rest for training

    print(f"\nDataset sizes:")
    print(f"  Training: {len(train_patients)} patients")
    print(f"  Validation: {len(val_patients)} patients")
    print(f"  Test (held out): {len(test_patients)} patients")

    # Create datasets
    train_dataset = OpenKBPDataset(
        train_patients,
        transform=get_transforms(training=True),
        include_dose=True,
    )
    val_dataset = OpenKBPDataset(
        val_patients,
        transform=get_transforms(training=False),
        include_dose=True,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Initialize model with timestamped run name
    model = DosePredictionModel(
        model_name=run_name,
        results_dir=results_dir,
        device=args.device,
        learning_rate=args.lr,
        num_filters=args.filters,
    )

    # Print model summary
    model_summary = model.get_model_summary()
    print(f"\nModel parameters: {model_summary['total_params']:,} total")
    print(f"Trainable parameters: {model_summary['trainable_params']:,}")
    print(f"Device: {model_summary['device']}")

    # Train
    print("\nStarting training...")
    model.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        save_frequency=args.save_freq,
        resume=not args.no_resume,
    )

    # Export training history
    exports_dir = model.results_dir / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)

    # Plot and save training history with custom visualization
    if model.train_losses and model.val_losses:
        plot_training_history(
            train_losses=model.train_losses,
            val_losses=model.val_losses,
            save_path=model.results_dir / "training_history.png",
            title=f"Training History - {run_name}",
        )

        # Export losses to CSV
        export_losses_csv(
            train_losses=model.train_losses,
            val_losses=model.val_losses,
            output_path=exports_dir / "losses.csv",
        )

    # Evaluation
    evaluation_results = {}
    if not args.skip_eval:
        # Load best model and make predictions on validation set
        print("\nGenerating predictions on validation set...")
        model.load_best_model()

        prediction_dir = model.results_dir / "validation-predictions"
        val_pred_dataset = OpenKBPDataset(
            val_patients,
            transform=get_inference_transforms(),
            include_dose=False,
        )
        val_pred_loader = DataLoader(
            val_pred_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
        )
        model.predict(val_pred_loader, prediction_dir)

        # Evaluate
        print("\nEvaluating predictions...")
        val_patient_ids = [p.stem for p in val_patients]
        evaluation_results = evaluate_predictions(
            pred_dir=prediction_dir,
            ref_dir=val_dir,
            patient_ids=val_patient_ids,
        )
        print_evaluation_results(evaluation_results)

        # Export evaluation results
        export_evaluation_results(
            results=evaluation_results,
            output_path=exports_dir / "evaluation_results.json",
            model_name=run_name,
            config=config,
        )

    # Export complete training summary
    export_training_summary(
        model_name=run_name,
        config=config,
        train_losses=model.train_losses,
        val_losses=model.val_losses,
        evaluation_results=evaluation_results,
        output_path=exports_dir / "training_summary.json",
        model_summary=model_summary,
    )

    # Run inference on held-out test set
    test_results = {}
    if test_patients and not args.skip_eval:
        print("\n" + "=" * 50)
        print("Running inference on held-out test set...")
        print("=" * 50)

        test_prediction_dir = model.results_dir / "test-predictions"
        test_pred_dataset = OpenKBPDataset(
            test_patients,
            transform=get_inference_transforms(),
            include_dose=False,
        )
        test_pred_loader = DataLoader(
            test_pred_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
        )
        model.predict(test_pred_loader, test_prediction_dir)

        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_patient_ids = [p.stem for p in test_patients]
        test_results = evaluate_predictions(
            pred_dir=test_prediction_dir,
            ref_dir=train_dir,  # Test patients came from train dir
            patient_ids=test_patient_ids,
        )
        print("\nTest Set Results:")
        print_evaluation_results(test_results)

        # Export test results
        export_evaluation_results(
            results=test_results,
            output_path=exports_dir / "test_results.json",
            model_name=run_name,
            config=config,
        )

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"\nResults saved to: {model.results_dir}")
    print(f"  - Model checkpoints: {model.model_dir}")
    print(f"  - Training history: {model.results_dir / 'training_history.png'}")
    print(f"  - Exports: {exports_dir}")
    if not args.skip_eval:
        print(f"  - Validation predictions: {prediction_dir}")
    if test_patients and not args.skip_eval:
        print(f"  - Test predictions: {test_prediction_dir}")


if __name__ == "__main__":
    main()
