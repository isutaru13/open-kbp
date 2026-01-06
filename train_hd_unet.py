#!/usr/bin/env python3
"""
HD U-Net training script for OpenKBP dose prediction.

This script trains a Hierarchical Dense U-Net for 3D radiation dose prediction.

Features:
- Dense blocks with bottleneck design for efficient feature reuse
- Attention gates for enhanced skip connections
- Deep supervision for improved gradient flow
- Mixed Precision Training (AMP)
- Gradient Accumulation
- OneCycleLR Scheduler

Usage:
    # Quick test
    python train_hd_unet.py --data-fraction 0.1 --epochs 5

    # Standard training (RTX 3060 12GB optimized)
    python train_hd_unet.py --epochs 200 --batch-size 2 --variant standard \\
        --scheduler onecycle --grad-accum 4 --amp --warmup-epochs 10

    # Lite model (faster, less memory)
    python train_hd_unet.py --epochs 200 --batch-size 4 --variant lite \\
        --scheduler onecycle --grad-accum 2 --amp

    # Large model (best quality, needs more memory)
    python train_hd_unet.py --epochs 300 --batch-size 1 --variant large \\
        --scheduler onecycle --grad-accum 8 --amp --grad-checkpoint
"""

import argparse
from datetime import datetime
from pathlib import Path

from torch.utils.data import DataLoader

from src import (
    DEFAULT_CONFIG,
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
from src.hd_unet_model import HDUNetDosePredictionModel
from src.visualization import plot_training_history


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train HD U-Net for OpenKBP dose prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model settings
    model_group = parser.add_argument_group("Model Settings")
    model_group.add_argument(
        "--model-name",
        type=str,
        default="hd_unet",
        help="Name for the model (used for saving)",
    )
    model_group.add_argument(
        "--variant",
        type=str,
        default="standard",
        choices=["lite", "standard", "large"],
        help="HD U-Net variant: lite (fast), standard (balanced), large (best)",
    )
    model_group.add_argument(
        "--init-features",
        type=int,
        default=48,
        help="Initial number of features (for custom config)",
    )
    model_group.add_argument(
        "--growth-rate",
        type=int,
        default=16,
        help="Dense block growth rate (for custom config)",
    )
    model_group.add_argument(
        "--no-attention",
        action="store_true",
        help="Disable attention gates in skip connections",
    )
    model_group.add_argument(
        "--no-deep-supervision",
        action="store_true",
        help="Disable deep supervision",
    )
    model_group.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate",
    )

    # Training settings
    train_group = parser.add_argument_group("Training Settings")
    train_group.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_CONFIG["num_epochs"],
        help="Number of training epochs",
    )
    train_group.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size per GPU (2 recommended for HD U-Net on 12GB)",
    )
    train_group.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    train_group.add_argument(
        "--lr-patience",
        type=int,
        default=5,
        help="Epochs to wait before reducing LR (plateau scheduler only)",
    )
    train_group.add_argument(
        "--warmup-epochs",
        type=int,
        default=10,
        help="Epochs before saving best model",
    )
    train_group.add_argument(
        "--min-epochs",
        type=int,
        default=0,
        help="Minimum epochs before early stopping",
    )

    # Optimization settings
    optim_group = parser.add_argument_group("Optimization Settings")
    optim_group.add_argument(
        "--amp",
        action="store_true",
        default=True,
        help="Enable Automatic Mixed Precision (FP16)",
    )
    optim_group.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable AMP (use FP32)",
    )
    optim_group.add_argument(
        "--grad-accum",
        type=int,
        default=2,
        help="Gradient accumulation steps",
    )
    optim_group.add_argument(
        "--scheduler",
        type=str,
        default="onecycle",
        choices=["plateau", "onecycle", "cosine"],
        help="LR scheduler type",
    )
    optim_group.add_argument(
        "--grad-checkpoint",
        action="store_true",
        help="Enable gradient checkpointing (saves VRAM)",
    )
    optim_group.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile (PyTorch 2.0+)",
    )
    optim_group.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Gradient clipping norm (0 to disable)",
    )

    # Augmentation settings
    aug_group = parser.add_argument_group("Augmentation Settings")
    aug_group.add_argument(
        "--augment-type",
        type=str,
        default="intensity",
        choices=["none", "intensity", "geometric", "full"],
        help="Augmentation strategy",
    )

    # Data settings
    data_group = parser.add_argument_group("Data Settings")
    data_group.add_argument(
        "--data-fraction",
        type=float,
        default=1.0,
        help="Fraction of training data to use (0.0-1.0)",
    )
    data_group.add_argument(
        "--test-split",
        type=float,
        default=0.1,
        help="Fraction of training data to hold out for testing",
    )
    data_group.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_CONFIG["num_workers"],
        help="Number of data loading workers",
    )
    data_group.add_argument(
        "--prefetch",
        type=int,
        default=2,
        help="Prefetch factor for data loading",
    )
    data_group.add_argument(
        "--persistent-workers",
        action="store_true",
        default=True,
        help="Keep data workers alive between epochs",
    )

    # Checkpointing
    ckpt_group = parser.add_argument_group("Checkpointing")
    ckpt_group.add_argument(
        "--save-freq",
        type=int,
        default=DEFAULT_CONFIG["save_frequency"],
        help="Save checkpoint every N epochs",
    )
    ckpt_group.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume training from latest checkpoint",
    )
    ckpt_group.add_argument(
        "--no-resume",
        action="store_true",
        help="Start training from scratch",
    )

    # Output settings
    output_group = parser.add_argument_group("Output Settings")
    output_group.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Base directory for results",
    )

    # Evaluation settings
    eval_group = parser.add_argument_group("Evaluation Settings")
    eval_group.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation after training",
    )

    return parser.parse_args()


def main() -> None:
    """Main training function."""
    args = parse_args()

    # Process boolean flags
    use_amp = args.amp and not args.no_amp
    resume = args.resume and not args.no_resume
    use_attention = not args.no_attention
    deep_supervision = not args.no_deep_supervision

    # Generate model name with timestamp if not specified
    if args.model_name == "hd_unet":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"hd_unet_{args.variant}_{timestamp}"
    else:
        model_name = args.model_name

    print("\n" + "=" * 60)
    print("HD U-Net Training for OpenKBP Dose Prediction")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Variant: {args.variant}")
    print(f"Attention Gates: {use_attention}")
    print(f"Deep Supervision: {deep_supervision}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Gradient Accumulation: {args.grad_accum}")
    print(f"Effective Batch Size: {args.batch_size * args.grad_accum}")
    print(f"Learning Rate: {args.lr}")
    print(f"AMP: {use_amp}")
    print(f"Augmentation: {args.augment_type}")
    print("=" * 60)

    # Data directories
    data_dir = Path("provided-data")
    train_dir = data_dir / "train-pats"
    val_dir = data_dir / "validation-pats"

    # Get patient directories
    train_patient_dirs = get_patient_dirs(train_dir)
    val_patient_dirs = get_patient_dirs(val_dir)

    # Apply data fraction
    if args.data_fraction < 1.0:
        num_train = max(1, int(len(train_patient_dirs) * args.data_fraction))
        train_patient_dirs = train_patient_dirs[:num_train]
        print(f"\nUsing {args.data_fraction * 100:.0f}% of training data: {num_train} patients")

    # Split off test set from training data
    if args.test_split > 0:
        num_test = max(1, int(len(train_patient_dirs) * args.test_split))
        test_patient_dirs = train_patient_dirs[-num_test:]
        train_patient_dirs = train_patient_dirs[:-num_test]
        print(f"Training: {len(train_patient_dirs)}, Test: {len(test_patient_dirs)}")
    else:
        test_patient_dirs = []

    print(f"\nDataset sizes:")
    print(f"  Training: {len(train_patient_dirs)} patients")
    print(f"  Validation: {len(val_patient_dirs)} patients")
    print(f"  Test: {len(test_patient_dirs)} patients")

    # Create transforms
    train_transform = get_transforms(augment_type=args.augment_type)
    val_transform = get_inference_transforms()

    # Create datasets
    train_dataset = OpenKBPDataset(
        patient_dirs=train_patient_dirs,
        transform=train_transform,
        include_dose=True,
    )
    val_dataset = OpenKBPDataset(
        patient_dirs=val_patient_dirs,
        transform=val_transform,
        include_dose=True,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=args.prefetch if args.num_workers > 0 else None,
        persistent_workers=args.persistent_workers and args.num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=args.prefetch if args.num_workers > 0 else None,
        persistent_workers=args.persistent_workers and args.num_workers > 0,
    )

    # Create model
    model = HDUNetDosePredictionModel(
        model_name=model_name,
        results_dir=args.results_dir,
        learning_rate=args.lr,
        model_variant=args.variant,
        lr_patience=args.lr_patience,
        warmup_epochs=args.warmup_epochs,
        min_epochs=args.min_epochs,
        init_features=args.init_features,
        growth_rate=args.growth_rate,
        use_attention=use_attention,
        deep_supervision=deep_supervision,
        dropout_rate=args.dropout,
        use_amp=use_amp,
        gradient_accumulation_steps=args.grad_accum,
        scheduler_type=args.scheduler,
        max_epochs=args.epochs,
        use_gradient_checkpointing=args.grad_checkpoint,
        use_compile=args.compile,
        grad_clip_norm=args.grad_clip if args.grad_clip > 0 else None,
    )

    # Print model summary
    summary = model.get_model_summary()
    print(f"\nModel Summary:")
    print(f"  Total Parameters: {summary['total_params']:,}")
    print(f"  Trainable Parameters: {summary['trainable_params']:,}")

    # Train
    print("\nStarting training...")
    model.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        save_frequency=args.save_freq,
        resume=resume,
    )

    # Export training results
    print("\nExporting training results...")
    export_losses_csv(
        model.train_losses,
        model.val_losses,
        model.results_dir / "losses.csv",
    )
    plot_training_history(
        model.train_losses,
        model.val_losses,
        model.results_dir / "training_history.png",
    )
    
    # Build config dict for export
    config = {
        "variant": args.variant,
        "init_features": args.init_features,
        "growth_rate": args.growth_rate,
        "use_attention": use_attention,
        "deep_supervision": deep_supervision,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "learning_rate": args.lr,
        "scheduler": args.scheduler,
        "augment_type": args.augment_type,
        "timing": model.get_timing_summary(),
    }
    
    export_training_summary(
        model_name=model_name,
        config=config,
        train_losses=model.train_losses,
        val_losses=model.val_losses,
        evaluation_results={},  # Will be filled after evaluation
        output_path=model.results_dir / "training_summary.json",
        model_summary=summary,
    )

    # Evaluation
    if not args.skip_eval and test_patient_dirs:
        print("\n" + "=" * 60)
        print("Evaluating on test set...")
        print("=" * 60)

        # Create test dataset and loader
        test_dataset = OpenKBPDataset(
            patient_dirs=test_patient_dirs,
            transform=val_transform,
            include_dose=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        # Load best model (or latest checkpoint as fallback) and predict
        if not model.load_best_model():
            # Fall back to latest checkpoint if best model wasn't saved
            # (e.g., warmup_epochs > num_epochs)
            print("Falling back to latest checkpoint...")
            model.load_checkpoint()
        pred_dir = model.results_dir / "predictions" / "test"
        model.predict(test_loader, pred_dir)

        # Evaluate - use train_dir as ref since test patients come from training set
        results = evaluate_predictions(
            pred_dir=pred_dir,
            ref_dir=train_dir,
        )
        print_evaluation_results(results)
        export_evaluation_results(results, model.results_dir / "evaluation_results.csv")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Results saved to: {model.results_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
