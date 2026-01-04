#!/usr/bin/env python3
"""
MONAI-based 3D U-Net training script for OpenKBP dose prediction.

Optimized for RTX 3060 12GB with:
- Mixed Precision Training (AMP) - ~1.5-2x speedup
- Gradient Accumulation - larger effective batch size
- OneCycleLR Scheduler - faster convergence
- Gradient Checkpointing - memory efficiency
- torch.compile support - PyTorch 2.0+ optimization

Usage:
    # Quick test
    python train_monai.py --data-fraction 0.1 --epochs 5

    # RTX 3060 12GB optimized (20 hour budget)
    python train_monai.py --epochs 400 --batch-size 4 --filters 64 --lr 3e-4 \\
        --scheduler onecycle --grad-accum 4 --amp --warmup-epochs 20

    # Maximum memory efficiency
    python train_monai.py --epochs 400 --batch-size 2 --filters 64 --lr 3e-4 \\
        --scheduler onecycle --grad-accum 8 --amp --grad-checkpoint
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
        description="Train MONAI 3D U-Net for OpenKBP dose prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model settings
    model_group = parser.add_argument_group("Model Settings")
    model_group.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_CONFIG["model_name"],
        help="Name for the model (used for saving)",
    )
    model_group.add_argument(
        "--filters",
        type=int,
        default=DEFAULT_CONFIG["num_filters"],
        help="Initial number of filters in U-Net (32/64/96)",
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
        default=DEFAULT_CONFIG["batch_size"],
        help="Batch size per GPU",
    )
    train_group.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_CONFIG["learning_rate"],
        help="Learning rate (base for OneCycleLR, max for others)",
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
        help="Epochs before saving best model (avoids early anomalies)",
    )
    train_group.add_argument(
        "--min-epochs",
        type=int,
        default=0,
        help="Minimum epochs before early stopping",
    )

    # Optimization settings (RTX 3060 optimizations)
    optim_group = parser.add_argument_group("Optimization Settings (RTX 3060)")
    optim_group.add_argument(
        "--amp",
        action="store_true",
        default=True,
        help="Enable Automatic Mixed Precision (FP16) - ~1.5-2x speedup",
    )
    optim_group.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable AMP (use FP32)",
    )
    optim_group.add_argument(
        "--grad-accum",
        type=int,
        default=1,
        help="Gradient accumulation steps (effective batch = batch-size × grad-accum)",
    )
    optim_group.add_argument(
        "--scheduler",
        type=str,
        default="onecycle",
        choices=["plateau", "onecycle", "cosine"],
        help="LR scheduler type (onecycle recommended for long training)",
    )
    optim_group.add_argument(
        "--grad-checkpoint",
        action="store_true",
        help="Enable gradient checkpointing (saves VRAM, slightly slower)",
    )
    optim_group.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile (PyTorch 2.0+, may speed up training)",
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
        help="Augmentation strategy: none (safest), intensity (recommended), geometric (may hurt!), full",
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
        "--no-resume",
        action="store_true",
        help="Start training from scratch (don't resume)",
    )
    ckpt_group.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation after training",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, default: auto)",
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Handle AMP flag
    use_amp = args.amp and not args.no_amp

    # Handle gradient clip
    grad_clip = args.grad_clip if args.grad_clip > 0 else None

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
        "lr_patience": args.lr_patience,
        "num_filters": args.filters,
        "save_frequency": args.save_freq,
        "num_workers": args.num_workers,
        "data_fraction": args.data_fraction,
        "test_split": args.test_split,
        "warmup_epochs": args.warmup_epochs,
        "min_epochs": args.min_epochs,
        # Optimization settings
        "use_amp": use_amp,
        "gradient_accumulation_steps": args.grad_accum,
        "scheduler_type": args.scheduler,
        "gradient_checkpointing": args.grad_checkpoint,
        "use_compile": args.compile,
        "grad_clip_norm": grad_clip,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "augment_type": args.augment_type,
    }

    print("=" * 70)
    print("OpenKBP Dose Prediction with MONAI 3D U-Net")
    print("Optimized for RTX 3060 12GB")
    print("=" * 70)

    print("\n📋 Run Configuration:")
    print(f"  Run name: {run_name}")
    print(f"  Model name: {args.model_name}")

    print("\n🏋️ Training Settings:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.grad_accum}")
    print(f"  Effective batch size: {args.batch_size * args.grad_accum}")
    print(f"  Learning rate: {args.lr}")
    print(f"  LR scheduler: {args.scheduler}")
    print(f"  Initial filters: {args.filters}")

    print("\n⚡ Optimizations:")
    print(f"  Mixed Precision (AMP): {use_amp}")
    print(f"  Gradient Checkpointing: {args.grad_checkpoint}")
    print(f"  torch.compile: {args.compile}")
    print(f"  Gradient clipping: {grad_clip}")
    print(f"  Augmentation: {args.augment_type}")

    print("\n📊 Data Settings:")
    print(f"  Data fraction: {args.data_fraction * 100:.0f}%")
    print(f"  Test split: {args.test_split * 100:.0f}%")
    print(f"  Num workers: {args.num_workers}")

    print("\n⏰ Training Control:")
    print(f"  Warmup epochs: {args.warmup_epochs}")
    print(f"  Save frequency: {args.save_freq}")
    print(f"  Resume: {not args.no_resume}")

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
        print(f"\n⚠️  Using {args.data_fraction * 100:.0f}% of data for quick testing")

    # Hold out test set from training data
    test_patients = []
    if args.test_split > 0:
        num_test = max(1, int(len(train_patients) * args.test_split))
        test_patients = train_patients[-num_test:]
        train_patients = train_patients[:-num_test]

    print("\n📁 Dataset Sizes:")
    print(f"  Training: {len(train_patients)} patients")
    print(f"  Validation: {len(val_patients)} patients")
    print(f"  Test (held out): {len(test_patients)} patients")

    # Create datasets
    print(f"\n🔄 Augmentation Strategy: {args.augment_type}")
    if args.augment_type == "geometric" or args.augment_type == "full":
        print("  ⚠️  WARNING: Geometric augmentation may hurt dose prediction!")
        print("  Consider using --augment-type intensity instead.")

    train_dataset = OpenKBPDataset(
        train_patients,
        transform=get_transforms(training=True, augment_type=args.augment_type),
        include_dose=True,
    )
    val_dataset = OpenKBPDataset(
        val_patients,
        transform=get_transforms(training=False),
        include_dose=True,
    )

    # DataLoader settings optimized for long training
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": True,
        "persistent_workers": args.persistent_workers and args.num_workers > 0,
        "prefetch_factor": args.prefetch if args.num_workers > 0 else None,
    }

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,  # Important for gradient accumulation
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    # Estimate training time
    # RTX 3060: ~45s/epoch with batch=4, filters=64 (FP32)
    # With AMP: ~25-30s/epoch
    estimated_epoch_time = 25 if use_amp else 45
    estimated_total_hours = (args.epochs * estimated_epoch_time) / 3600
    print("\n⏱️  Estimated Training Time:")
    print(f"  Per epoch: ~{estimated_epoch_time}s")
    print(f"  Total: ~{estimated_total_hours:.1f} hours")

    # Initialize model with all optimization settings
    model = DosePredictionModel(
        model_name=run_name,
        results_dir=results_dir,
        device=args.device,
        learning_rate=args.lr,
        num_filters=args.filters,
        lr_patience=args.lr_patience,
        warmup_epochs=args.warmup_epochs,
        min_epochs=args.min_epochs,
        # Optimization settings
        use_amp=use_amp,
        gradient_accumulation_steps=args.grad_accum,
        scheduler_type=args.scheduler,
        max_epochs=args.epochs,
        use_gradient_checkpointing=args.grad_checkpoint,
        use_compile=args.compile,
        grad_clip_norm=grad_clip,
    )

    # Print model summary
    model_summary = model.get_model_summary()
    print("\n🧠 Model Summary:")
    print(f"  Parameters: {model_summary['total_params']:,} total")
    print(f"  Trainable: {model_summary['trainable_params']:,}")
    print(f"  Device: {model_summary['device']}")

    # Train
    print("\n" + "=" * 70)
    print("🚀 Starting Training...")
    print("=" * 70)

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

    # Plot and save training history
    if model.train_losses and model.val_losses:
        plot_training_history(
            train_losses=model.train_losses,
            val_losses=model.val_losses,
            save_path=model.results_dir / "training_history.png",
            title=f"Training History - {run_name}",
        )

        export_losses_csv(
            train_losses=model.train_losses,
            val_losses=model.val_losses,
            output_path=exports_dir / "losses.csv",
        )

    # Evaluation
    evaluation_results = {}
    prediction_dir = None
    if not args.skip_eval:
        print("\n" + "=" * 70)
        print("📊 Generating predictions on validation set...")
        print("=" * 70)

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
        print("\n📈 Evaluating predictions...")
        val_patient_ids = [p.stem for p in val_patients]
        evaluation_results = evaluate_predictions(
            pred_dir=prediction_dir,
            ref_dir=val_dir,
            patient_ids=val_patient_ids,
        )
        print_evaluation_results(evaluation_results)

        export_evaluation_results(
            results=evaluation_results,
            output_path=exports_dir / "evaluation_results.json",
            model_name=run_name,
            config=config,
        )

    # Export complete training summary
    timing_summary = model.get_timing_summary()
    export_training_summary(
        model_name=run_name,
        config=config,
        train_losses=model.train_losses,
        val_losses=model.val_losses,
        evaluation_results=evaluation_results,
        output_path=exports_dir / "training_summary.json",
        model_summary={**model_summary, "timing": timing_summary},
    )

    # Print timing summary
    if timing_summary:
        print("\n⏱️  Timing Summary:")
        print(f"  Total training time: {timing_summary['total_time_hours']:.2f} hours")
        print(f"  Average epoch time: {timing_summary['avg_epoch_time']:.1f} seconds")
        print(f"  Min epoch time: {timing_summary['min_epoch_time']:.1f} seconds")
        print(f"  Max epoch time: {timing_summary['max_epoch_time']:.1f} seconds")

    # Run inference on held-out test set
    test_results = {}
    test_prediction_dir = None
    if test_patients and not args.skip_eval:
        print("\n" + "=" * 70)
        print("🧪 Running inference on held-out test set...")
        print("=" * 70)

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

        print("\n📈 Evaluating on test set...")
        test_patient_ids = [p.stem for p in test_patients]
        test_results = evaluate_predictions(
            pred_dir=test_prediction_dir,
            ref_dir=train_dir,
            patient_ids=test_patient_ids,
        )
        print("\n🧪 Test Set Results:")
        print_evaluation_results(test_results)

        export_evaluation_results(
            results=test_results,
            output_path=exports_dir / "test_results.json",
            model_name=run_name,
            config=config,
        )

    # Final summary
    print("\n" + "=" * 70)
    print("✅ Training Complete!")
    print("=" * 70)
    print(f"\n📁 Results saved to: {model.results_dir}")
    print(f"  📦 Model checkpoints: {model.model_dir}")
    print(f"  📊 Training history: {model.results_dir / 'training_history.png'}")
    print(f"  📄 Exports: {exports_dir}")
    if prediction_dir is not None:
        print(f"  🔮 Validation predictions: {prediction_dir}")
    if test_prediction_dir is not None:
        print(f"  🧪 Test predictions: {test_prediction_dir}")

    # Print recommended next steps
    print("\n" + "=" * 70)
    print("💡 Recommended Commands for RTX 3060 12GB (20 hour budget):")
    print("=" * 70)
    print("""
# Option 1: Balanced (recommended)
python train_monai.py --epochs 400 --batch-size 4 --filters 64 --lr 3e-4 \\
    --scheduler onecycle --grad-accum 4 --warmup-epochs 20 --save-freq 50

# Option 2: Maximum epochs (slower convergence)
python train_monai.py --epochs 600 --batch-size 4 --filters 48 --lr 2e-4 \\
    --scheduler onecycle --grad-accum 4 --warmup-epochs 30 --save-freq 100

# Option 3: Larger model (fewer epochs)
python train_monai.py --epochs 300 --batch-size 2 --filters 96 --lr 3e-4 \\
    --scheduler onecycle --grad-accum 8 --warmup-epochs 15 --save-freq 50

# Option 4: No augmentation (test if augmentation is hurting results)
python train_monai.py --epochs 400 --batch-size 4 --filters 64 --lr 3e-4 \\
    --scheduler onecycle --grad-accum 4 --warmup-epochs 20 --augment-type none
""")


if __name__ == "__main__":
    main()
