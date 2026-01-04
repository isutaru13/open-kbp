"""
MONAI-based dose prediction model and trainer.

This module provides the DosePredictionModel class for training
and inference using MONAI's 3D U-Net architecture.
"""

import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from monai.networks.nets import UNet
from torch.utils.data import DataLoader
from tqdm import tqdm

from .constants import NUM_INPUT_CHANNELS


class MaskedMAELoss(torch.nn.Module):
    """Masked Mean Absolute Error Loss.

    Computes MAE only in regions where dose deposition is possible,
    as defined by the possible_dose_mask.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute masked MAE loss.

        Args:
            pred: Predicted dose (B, 1, H, W, D)
            target: Target dose (B, 1, H, W, D)
            mask: Possible dose mask (B, 1, H, W, D)

        Returns:
            Scalar loss value
        """
        # Apply mask
        masked_pred = pred * mask
        masked_target = target * mask

        # Compute MAE only on masked regions
        abs_diff = torch.abs(masked_pred - masked_target)
        loss = abs_diff.sum() / (mask.sum() + 1e-8)
        return loss


class DosePredictionModel:
    """MONAI-based dose prediction model trainer.

    This class handles model creation, training, validation,
    checkpointing, and inference for dose prediction.

    Attributes:
        model_name: Name identifier for the model
        results_dir: Directory to save results
        model_dir: Directory to save model checkpoints
        device: PyTorch device (cuda/cpu)
        model: MONAI UNet model
        criterion: Loss function
        optimizer: Adam optimizer
        scheduler: Learning rate scheduler
    """

    def __init__(
        self,
        model_name: str = "monai_unet",
        results_dir: Path = Path("results"),
        device: Optional[str] = None,
        learning_rate: float = 1e-4,
        num_filters: int = 32,
        lr_patience: int = 5,
    ):
        """
        Initialize the dose prediction model.

        Args:
            model_name: Name for the model (used for saving)
            results_dir: Base directory for results
            device: Device to use ('cuda', 'cpu', or None for auto)
            learning_rate: Learning rate for optimizer
            num_filters: Initial number of filters in U-Net
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

        # Build MONAI UNet
        self.model = UNet(
            spatial_dims=3,
            in_channels=NUM_INPUT_CHANNELS,
            out_channels=1,
            channels=(
                num_filters,
                num_filters * 2,
                num_filters * 4,
                num_filters * 8,
                num_filters * 16,
            ),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm="batch",
            dropout=0.2,
        ).to(self.device)

        # Loss and optimizer
        self.criterion = MaskedMAELoss()
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Learning rate scheduler - reduce on plateau with faster response
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,  # Halve LR when plateau
            patience=lr_patience,  # Wait N epochs before reducing
            min_lr=1e-7,  # Don't go below this
            threshold=0.01,  # Minimum improvement to count as progress
        )

        # Training history
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.epoch_times: list[float] = []
        self.current_epoch = 0

    def save_checkpoint(self, epoch: int, val_loss: float) -> None:
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch number
            val_loss: Validation loss for this epoch
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "epoch_times": self.epoch_times,
            "val_loss": val_loss,
        }
        torch.save(checkpoint, self.model_dir / f"epoch_{epoch}.pt")
        print(f"Saved checkpoint at epoch {epoch}")

    def load_checkpoint(self, epoch: Optional[int] = None) -> None:
        """
        Load model checkpoint.

        Args:
            epoch: Specific epoch to load. If None, loads the latest.
        """
        checkpoints = list(self.model_dir.glob("epoch_*.pt"))
        if not checkpoints:
            print("No checkpoints found, starting from scratch")
            return

        if epoch is None:
            # Find latest checkpoint
            epochs = [int(c.stem.split("_")[1]) for c in checkpoints]
            epoch = max(epochs)

        checkpoint_path = self.model_dir / f"epoch_{epoch}.pt"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.train_losses = checkpoint["train_losses"]
            self.val_losses = checkpoint["val_losses"]
            self.epoch_times = checkpoint.get("epoch_times", [])
            self.current_epoch = checkpoint["epoch"]
            print(f"Loaded checkpoint from epoch {epoch}")
        else:
            print(f"Checkpoint for epoch {epoch} not found")

    def load_best_model(self) -> None:
        """Load the best model based on validation loss."""
        best_model_path = self.model_dir / "best_model.pt"
        if best_model_path.exists():
            self.model.load_state_dict(
                torch.load(best_model_path, map_location=self.device)
            )
            print("Loaded best model")
        else:
            print("Best model not found")

    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.

        Args:
            train_loader: DataLoader for training data

        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            image = batch["image"].to(self.device)
            dose = batch["dose"].to(self.device)
            mask = batch["mask"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            pred = self.model(image)

            # Apply ReLU to ensure non-negative dose predictions
            pred = torch.relu(pred)

            # Compute loss
            loss = self.criterion(pred, dose, mask)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / num_batches

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate the model.

        Args:
            val_loader: DataLoader for validation data

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(val_loader, desc="Validation")
        for batch in progress_bar:
            image = batch["image"].to(self.device)
            dose = batch["dose"].to(self.device)
            mask = batch["mask"].to(self.device)

            # Forward pass
            pred = self.model(image)
            pred = torch.relu(pred)

            # Compute loss
            loss = self.criterion(pred, dose, mask)

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
        """
        Train the model.

        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs: Total number of epochs to train
            save_frequency: Save checkpoint every N epochs
            resume: Whether to resume from latest checkpoint
        """
        if resume:
            self.load_checkpoint()

        best_val_loss = float("inf")
        start_epoch = self.current_epoch

        for epoch in range(start_epoch, num_epochs):
            epoch_start_time = time.time()
            self.current_epoch = epoch
            print(f"\n{'=' * 50}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'=' * 50}")

            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            # Validate
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)

            # Update learning rate scheduler
            self.scheduler.step(val_loss)

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Record epoch time
            epoch_time = time.time() - epoch_start_time
            self.epoch_times.append(epoch_time)

            # Calculate ETA
            avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
            remaining_epochs = num_epochs - (epoch + 1)
            eta_seconds = avg_epoch_time * remaining_epochs
            eta_min, eta_sec = divmod(int(eta_seconds), 60)
            eta_hr, eta_min = divmod(eta_min, 60)

            print(
                f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, "
                f"Val Loss = {val_loss:.4f}, "
                f"LR = {current_lr:.2e}, "
                f"Time = {epoch_time:.1f}s, "
                f"ETA = {eta_hr}h {eta_min}m {eta_sec}s"
            )

            # Save checkpoint
            if (epoch + 1) % save_frequency == 0:
                self.save_checkpoint(epoch + 1, val_loss)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.model_dir / "best_model.pt")
                print(f"New best model saved with val_loss = {val_loss:.4f}")

        # Save final model
        self.save_checkpoint(num_epochs, val_loss)

    @torch.no_grad()
    def predict(self, data_loader: DataLoader, output_dir: Path) -> None:
        """
        Generate predictions for a dataset.

        Args:
            data_loader: DataLoader for inference data
            output_dir: Directory to save predictions
        """
        self.model.eval()
        output_dir.mkdir(parents=True, exist_ok=True)

        progress_bar = tqdm(data_loader, desc="Predicting")
        for batch in progress_bar:
            image = batch["image"].to(self.device)
            mask = batch["mask"].to(self.device)
            patient_ids = batch["patient_id"]

            # Forward pass
            pred = self.model(image)
            pred = torch.relu(pred)

            # Apply mask
            pred = pred * mask

            # Save predictions
            pred_np = pred.cpu().numpy()
            for i, patient_id in enumerate(patient_ids):
                dose_pred = pred_np[i, 0]  # (128, 128, 128)
                self._save_dose_prediction(dose_pred, patient_id, output_dir)

    def _save_dose_prediction(
        self, dose: np.ndarray, patient_id: str, output_dir: Path
    ) -> None:
        """
        Save dose prediction in sparse CSV format.

        Args:
            dose: 3D dose array
            patient_id: Patient identifier
            output_dir: Directory to save the prediction
        """
        flat_dose = dose.flatten()
        nonzero_mask = flat_dose > 0
        indices = np.where(nonzero_mask)[0]
        values = flat_dose[nonzero_mask]

        df = pd.DataFrame({"data": values}, index=indices)
        df.to_csv(output_dir / f"{patient_id}.csv")

    def get_model_summary(self) -> dict:
        """
        Get model summary information.

        Returns:
            Dictionary with model parameter counts
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "device": str(self.device),
        }

    def get_timing_summary(self) -> dict:
        """
        Get timing summary information.

        Returns:
            Dictionary with timing statistics
        """
        if not self.epoch_times:
            return {}

        return {
            "total_time_seconds": sum(self.epoch_times),
            "total_time_minutes": sum(self.epoch_times) / 60,
            "avg_epoch_time": sum(self.epoch_times) / len(self.epoch_times),
            "min_epoch_time": min(self.epoch_times),
            "max_epoch_time": max(self.epoch_times),
            "epoch_times": self.epoch_times,
        }
