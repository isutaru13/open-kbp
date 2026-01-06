"""
HD U-Net based dose prediction model and trainer.

This module provides the HDUNetDosePredictionModel class for training
and inference using the HD U-Net architecture.

Optimized for RTX 3060 12GB with:
- Mixed Precision Training (AMP)
- Gradient Accumulation
- OneCycleLR Scheduler
- Gradient Checkpointing
- Deep Supervision support
- torch.compile support
"""

import time
from pathlib import Path
from typing import List, Literal, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from .constants import NUM_INPUT_CHANNELS
from .hd_unet import HDUNet, HDUNetLite, get_hd_unet
from .losses import MaskedMAELoss


class DeepSupervisionLoss(torch.nn.Module):
    """Loss function with deep supervision support.
    
    Combines main output loss with weighted auxiliary losses.
    
    Args:
        base_loss: Base loss function (e.g., MaskedMAELoss)
        weights: Weights for auxiliary outputs (from deepest to shallowest)
    """
    
    def __init__(
        self,
        base_loss: torch.nn.Module,
        weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.base_loss = base_loss
        self.weights = weights or [0.5, 0.25, 0.125]  # Decreasing weights for deeper levels
    
    def forward(
        self,
        pred: Union[torch.Tensor, tuple],
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss with optional deep supervision.
        
        Args:
            pred: Main prediction or tuple of (main_pred, [aux_preds])
            target: Target tensor
            mask: Mask tensor
            
        Returns:
            Total loss
        """
        if isinstance(pred, tuple):
            main_pred, aux_preds = pred
            
            # Main loss
            main_loss = self.base_loss(main_pred, target, mask)
            
            # Auxiliary losses
            aux_loss = 0.0
            for i, aux_pred in enumerate(aux_preds):
                weight = self.weights[i] if i < len(self.weights) else 0.1
                aux_loss += weight * self.base_loss(aux_pred, target, mask)
            
            return main_loss + aux_loss
        else:
            return self.base_loss(pred, target, mask)


class HDUNetDosePredictionModel:
    """HD U-Net based dose prediction model trainer.

    This class handles model creation, training, validation,
    checkpointing, and inference for dose prediction using HD U-Net.

    Optimized for memory-constrained GPUs like RTX 3060 12GB.

    Attributes:
        model_name: Name identifier for the model
        results_dir: Directory to save results
        model_dir: Directory to save model checkpoints
        device: PyTorch device (cuda/cpu)
        model: HD U-Net model
        criterion: Loss function
        optimizer: AdamW optimizer
        scheduler: Learning rate scheduler
    """

    def __init__(
        self,
        model_name: str = "hd_unet",
        results_dir: Path = Path("results"),
        device: Optional[str] = None,
        learning_rate: float = 1e-4,
        model_variant: Literal["lite", "standard", "large"] = "standard",
        lr_patience: int = 5,
        warmup_epochs: int = 0,
        min_epochs: int = 0,
        # HD U-Net specific settings
        init_features: int = 48,
        growth_rate: int = 16,
        use_attention: bool = True,
        deep_supervision: bool = True,
        dropout_rate: float = 0.2,
        # Optimization settings
        use_amp: bool = True,
        gradient_accumulation_steps: int = 1,
        scheduler_type: Literal["plateau", "onecycle", "cosine"] = "plateau",
        max_epochs: int = 100,
        use_gradient_checkpointing: bool = False,
        use_compile: bool = False,
        grad_clip_norm: Optional[float] = 1.0,
    ):
        """
        Initialize the HD U-Net dose prediction model.

        Args:
            model_name: Name for the model (used for saving)
            results_dir: Base directory for results
            device: Device to use ('cuda', 'cpu', or None for auto)
            learning_rate: Learning rate for optimizer
            model_variant: HD U-Net variant ('lite', 'standard', 'large')
            lr_patience: Epochs to wait before reducing LR on plateau
            warmup_epochs: Number of epochs before saving best model
            min_epochs: Minimum epochs before early stopping is allowed
            init_features: Initial number of features
            growth_rate: Dense block growth rate
            use_attention: Use attention gates in skip connections
            deep_supervision: Enable deep supervision during training
            dropout_rate: Dropout rate
            use_amp: Enable Automatic Mixed Precision (FP16) training
            gradient_accumulation_steps: Number of steps to accumulate gradients
            scheduler_type: LR scheduler type ('plateau', 'onecycle', 'cosine')
            max_epochs: Maximum epochs (needed for onecycle/cosine schedulers)
            use_gradient_checkpointing: Enable gradient checkpointing to save VRAM
            use_compile: Use torch.compile for faster execution (PyTorch 2.0+)
            grad_clip_norm: Max norm for gradient clipping (None to disable)
        """
        self.model_name = model_name
        self.results_dir = results_dir / model_name
        self.model_dir = self.results_dir / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"Using device: {self.device}")

        # Optimization settings
        self.use_amp = use_amp and self.device.type == "cuda"
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.scheduler_type = scheduler_type
        self.max_epochs = max_epochs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.grad_clip_norm = grad_clip_norm
        self.deep_supervision = deep_supervision

        # Print optimization settings
        print(f"\n{'=' * 50}")
        print("HD U-Net Configuration")
        print("=" * 50)
        print(f"Model Variant: {model_variant}")
        print(f"Init Features: {init_features}")
        print(f"Growth Rate: {growth_rate}")
        print(f"Use Attention: {use_attention}")
        print(f"Deep Supervision: {deep_supervision}")
        print(f"Dropout Rate: {dropout_rate}")
        print(f"\nOptimization Settings:")
        print(f"Mixed Precision (AMP): {self.use_amp}")
        print(f"Gradient Accumulation Steps: {self.gradient_accumulation_steps}")
        print(f"Scheduler Type: {self.scheduler_type}")
        print(f"Gradient Checkpointing: {self.use_gradient_checkpointing}")
        print(f"Gradient Clip Norm: {self.grad_clip_norm}")
        print("=" * 50)

        # Build HD U-Net
        self.model_variant = model_variant
        self.init_features = init_features
        self.growth_rate = growth_rate
        
        if model_variant in ["lite", "standard", "large"]:
            self.model = get_hd_unet(
                variant=model_variant,
                in_channels=NUM_INPUT_CHANNELS,
                out_channels=1,
                use_checkpoint=use_gradient_checkpointing,
                use_attention=use_attention,
                deep_supervision=deep_supervision,
                dropout_rate=dropout_rate,
            )
        else:
            # Custom configuration
            self.model = HDUNet(
                in_channels=NUM_INPUT_CHANNELS,
                out_channels=1,
                init_features=init_features,
                growth_rate=growth_rate,
                use_attention=use_attention,
                deep_supervision=deep_supervision,
                dropout_rate=dropout_rate,
                use_checkpoint=use_gradient_checkpointing,
            )
        
        self.model = self.model.to(self.device)

        # Print model stats
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\nModel Parameters: {total_params:,} total, {trainable_params:,} trainable")

        # Compile model if requested (PyTorch 2.0+)
        if use_compile:
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("Model compiled with torch.compile")
            except Exception as e:
                print(f"torch.compile not available: {e}")

        # Loss function with deep supervision support
        base_loss = MaskedMAELoss()
        if deep_supervision:
            self.criterion = DeepSupervisionLoss(base_loss)
        else:
            self.criterion = base_loss
        
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5,
            betas=(0.9, 0.999),
        )

        # Mixed precision scaler
        self.scaler = GradScaler('cuda', enabled=self.use_amp)

        # Learning rate scheduler
        self.scheduler = None
        self.lr_patience = lr_patience
        self._init_scheduler(scheduler_type, max_epochs)

        # Training history
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.learning_rates: list[float] = []
        self.epoch_times: list[float] = []
        self.current_epoch = 0

        # Long training settings
        self.warmup_epochs = warmup_epochs
        self.min_epochs = min_epochs

    def _init_scheduler(
        self,
        scheduler_type: str,
        max_epochs: int,
        steps_per_epoch: int = 1,
    ) -> None:
        """Initialize learning rate scheduler."""
        if scheduler_type == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=self.lr_patience,
                min_lr=1e-7,
                threshold=0.01,
            )
        elif scheduler_type == "onecycle":
            self.scheduler = None  # Initialized in train() with correct steps
        elif scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max_epochs,
                eta_min=1e-7,
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    def save_checkpoint(self, epoch: int, val_loss: float) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "learning_rates": self.learning_rates,
            "epoch_times": self.epoch_times,
            "val_loss": val_loss,
            "model_config": {
                "variant": self.model_variant,
                "init_features": self.init_features,
                "growth_rate": self.growth_rate,
                "deep_supervision": self.deep_supervision,
            },
        }
        if self.scheduler is not None and self.scheduler_type != "onecycle":
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, self.model_dir / f"epoch_{epoch}.pt")
        print(f"Saved checkpoint at epoch {epoch}")

    def load_checkpoint(self, epoch: Optional[int] = None) -> None:
        """Load model checkpoint."""
        checkpoints = list(self.model_dir.glob("epoch_*.pt"))
        if not checkpoints:
            print("No checkpoints found, starting from scratch")
            return

        if epoch is None:
            epochs = [int(c.stem.split("_")[1]) for c in checkpoints]
            epoch = max(epochs)

        checkpoint_path = self.model_dir / f"epoch_{epoch}.pt"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scaler_state_dict" in checkpoint:
                self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            if (
                "scheduler_state_dict" in checkpoint
                and self.scheduler is not None
                and self.scheduler_type != "onecycle"
            ):
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.train_losses = checkpoint["train_losses"]
            self.val_losses = checkpoint["val_losses"]
            self.learning_rates = checkpoint.get("learning_rates", [])
            self.epoch_times = checkpoint.get("epoch_times", [])
            self.current_epoch = checkpoint["epoch"]
            print(f"Loaded checkpoint from epoch {epoch}")
        else:
            print(f"Checkpoint for epoch {epoch} not found")

    def load_best_model(self) -> bool:
        """Load the best model based on validation loss.
        
        Returns:
            True if best model was loaded, False otherwise.
        """
        best_model_path = self.model_dir / "best_model.pt"
        if best_model_path.exists():
            self.model.load_state_dict(
                torch.load(best_model_path, map_location=self.device)
            )
            print("Loaded best model")
            return True
        else:
            print("Best model not found")
            return False

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch with AMP and gradient accumulation."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        self.optimizer.zero_grad()

        progress_bar = tqdm(train_loader, desc="Training")
        for batch_idx, batch in enumerate(progress_bar):
            image = batch["image"].to(self.device, non_blocking=True)
            dose = batch["dose"].to(self.device, non_blocking=True)
            mask = batch["mask"].to(self.device, non_blocking=True)

            # Forward pass with AMP
            with autocast('cuda', enabled=self.use_amp):
                pred = self.model(image)
                
                # Apply ReLU to main output (handle deep supervision tuple)
                if isinstance(pred, tuple):
                    main_pred, aux_preds = pred
                    main_pred = torch.relu(main_pred)
                    aux_preds = [torch.relu(aux) for aux in aux_preds]
                    pred = (main_pred, aux_preds)
                else:
                    pred = torch.relu(pred)
                
                loss = self.criterion(pred, dose, mask)
                loss = loss / self.gradient_accumulation_steps

            # Backward pass
            self.scaler.scale(loss).backward()


            # Update weights after accumulation steps
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.grad_clip_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip_norm
                    )

                # Track scale before step to detect if optimizer step was skipped
                old_scale = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                new_scale = self.scaler.get_scale()
                
                # Only step scheduler if optimizer.step() was actually called
                # (scale stays same or increases when step is skipped due to inf/nan)
                if self.scheduler_type == "onecycle" and self.scheduler is not None:
                    if old_scale <= new_scale:
                        # Optimizer step was skipped, don't step scheduler
                        pass
                    else:
                        self.scheduler.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
            progress_bar.set_postfix(
                {"loss": f"{loss.item() * self.gradient_accumulation_steps:.4f}"}
            )


        # Handle remaining gradients
        if num_batches % self.gradient_accumulation_steps != 0:
            if self.grad_clip_norm is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_norm
                )
            old_scale = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            new_scale = self.scaler.get_scale()
            if self.scheduler_type == "onecycle" and self.scheduler is not None:
                if old_scale > new_scale:  # Only step if optimizer actually stepped
                    self.scheduler.step()
            self.optimizer.zero_grad()

        return total_loss / num_batches

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model with AMP."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        # Use base loss for validation (no deep supervision)
        base_loss = MaskedMAELoss()

        progress_bar = tqdm(val_loader, desc="Validation")
        for batch in progress_bar:
            image = batch["image"].to(self.device, non_blocking=True)
            dose = batch["dose"].to(self.device, non_blocking=True)
            mask = batch["mask"].to(self.device, non_blocking=True)

            with autocast('cuda', enabled=self.use_amp):
                pred = self.model(image)
                
                # Handle deep supervision output (only use main output for validation)
                if isinstance(pred, tuple):
                    pred = pred[0]
                
                pred = torch.relu(pred)
                loss = base_loss(pred, dose, mask)

            total_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / num_batches

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 100,
        save_frequency: int = 5,
        resume: bool = True,
    ) -> None:
        """Train the model."""
        if resume:
            self.load_checkpoint()

        # Set start_epoch BEFORE scheduler initialization so we know remaining epochs
        start_epoch = self.current_epoch
        
        # Initialize OneCycleLR scheduler AFTER loading checkpoint
        # so we know the correct start_epoch
        if self.scheduler_type == "onecycle":
            steps_per_epoch = len(train_loader) // self.gradient_accumulation_steps
            # Account for any remaining batch that doesn't fill accumulation steps
            if len(train_loader) % self.gradient_accumulation_steps != 0:
                steps_per_epoch += 1
            remaining_epochs = num_epochs - start_epoch
            total_steps = steps_per_epoch * remaining_epochs
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.learning_rate * 10,
                total_steps=total_steps,
                pct_start=0.1,
                anneal_strategy="cos",
                div_factor=10,
                final_div_factor=100,
            )
            print(
                f"OneCycleLR initialized: {total_steps} total steps "
                f"({steps_per_epoch} steps/epoch × {remaining_epochs} remaining epochs), "
                f"max_lr={self.learning_rate * 10:.2e}"
            )

        best_val_loss = float("inf")

        effective_batch_size = (
            train_loader.batch_size * self.gradient_accumulation_steps
        )
        print(
            f"\nEffective batch size: {effective_batch_size} "
            f"(batch={train_loader.batch_size} × accum={self.gradient_accumulation_steps})"
        )

        for epoch in range(start_epoch, num_epochs):
            epoch_start_time = time.time()
            self.current_epoch = epoch
            print(f"\n{'=' * 50}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print("=" * 50)

            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)


            # Update scheduler (plateau/cosine only at end of epoch)
            if self.scheduler_type == "plateau":
                self.scheduler.step(val_loss)
            elif self.scheduler_type == "cosine":
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.learning_rates.append(current_lr)

            epoch_time = time.time() - epoch_start_time
            self.epoch_times.append(epoch_time)

            avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
            remaining_epochs = num_epochs - (epoch + 1)
            eta_seconds = avg_epoch_time * remaining_epochs
            eta_min, eta_sec = divmod(int(eta_seconds), 60)
            eta_hr, eta_min = divmod(eta_min, 60)

            if self.device.type == "cuda":
                mem_used = torch.cuda.max_memory_allocated() / 1024**3
                mem_str = f", VRAM = {mem_used:.1f}GB"
                torch.cuda.reset_peak_memory_stats()
            else:
                mem_str = ""

            print(
                f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, "
                f"Val Loss = {val_loss:.4f}, "
                f"LR = {current_lr:.2e}, "
                f"Time = {epoch_time:.1f}s{mem_str}, "
                f"ETA = {eta_hr}h {eta_min}m {eta_sec}s"
            )

            if (epoch + 1) % save_frequency == 0:
                self.save_checkpoint(epoch + 1, val_loss)

            if val_loss < best_val_loss and (epoch + 1) > self.warmup_epochs:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.model_dir / "best_model.pt")
                print(f"New best model saved with val_loss = {val_loss:.4f}")
            elif val_loss < best_val_loss and (epoch + 1) <= self.warmup_epochs:
                print(
                    f"Warmup period ({epoch + 1}/{self.warmup_epochs}): "
                    f"skipping best model save (val_loss = {val_loss:.4f})"
                )

        self.save_checkpoint(num_epochs, val_loss)

    @torch.no_grad()
    def predict(self, data_loader: DataLoader, output_dir: Path) -> None:
        """Generate predictions for a dataset."""
        self.model.eval()
        output_dir.mkdir(parents=True, exist_ok=True)

        progress_bar = tqdm(data_loader, desc="Predicting")
        for batch in progress_bar:
            image = batch["image"].to(self.device, non_blocking=True)
            mask = batch["mask"].to(self.device, non_blocking=True)
            patient_ids = batch["patient_id"]

            with autocast('cuda', enabled=self.use_amp):
                pred = self.model(image)
                
                # Handle deep supervision output
                if isinstance(pred, tuple):
                    pred = pred[0]
                
                pred = torch.relu(pred)

            pred = pred * mask
            pred_np = pred.float().cpu().numpy()
            
            for i, patient_id in enumerate(patient_ids):
                dose_pred = pred_np[i, 0]
                self._save_dose_prediction(dose_pred, patient_id, output_dir)

    def _save_dose_prediction(
        self, dose: np.ndarray, patient_id: str, output_dir: Path
    ) -> None:
        """Save dose prediction in sparse CSV format."""
        flat_dose = dose.flatten()
        nonzero_mask = flat_dose > 0
        indices = np.where(nonzero_mask)[0]
        values = flat_dose[nonzero_mask]

        df = pd.DataFrame({"data": values}, index=indices)
        df.to_csv(output_dir / f"{patient_id}.csv")

    def get_model_summary(self) -> dict:
        """Get model summary information."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "device": str(self.device),
            "model_variant": self.model_variant,
            "init_features": self.init_features,
            "growth_rate": self.growth_rate,
            "deep_supervision": self.deep_supervision,
            "use_amp": self.use_amp,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "scheduler_type": self.scheduler_type,
        }

    def get_timing_summary(self) -> dict:
        """Get timing summary information."""
        if not self.epoch_times:
            return {}

        return {
            "total_time_seconds": sum(self.epoch_times),
            "total_time_minutes": sum(self.epoch_times) / 60,
            "total_time_hours": sum(self.epoch_times) / 3600,
            "avg_epoch_time": sum(self.epoch_times) / len(self.epoch_times),
            "min_epoch_time": min(self.epoch_times),
            "max_epoch_time": max(self.epoch_times),
            "epoch_times": self.epoch_times,
        }
