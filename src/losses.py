"""
Custom loss functions for OpenKBP dose prediction.
"""

import torch
import torch.nn as nn


class MaskedMAELoss(nn.Module):
    """
    Masked Mean Absolute Error Loss.

    Computes MAE only in regions where dose deposition is possible,
    as defined by the possible_dose_mask.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute masked MAE loss.

        Args:
            pred: Predicted dose tensor (B, 1, H, W, D)
            target: Target dose tensor (B, 1, H, W, D)
            mask: Possible dose mask tensor (B, 1, H, W, D)

        Returns:
            Scalar loss value
        """
        # Apply mask to both predictions and targets
        masked_pred = pred * mask
        masked_target = target * mask

        # Compute MAE only on masked regions
        abs_diff = torch.abs(masked_pred - masked_target)
        loss = abs_diff.sum() / (mask.sum() + 1e-8)

        return loss


class MaskedMSELoss(nn.Module):
    """
    Masked Mean Squared Error Loss.

    Computes MSE only in regions where dose deposition is possible,
    as defined by the possible_dose_mask.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute masked MSE loss.

        Args:
            pred: Predicted dose tensor (B, 1, H, W, D)
            target: Target dose tensor (B, 1, H, W, D)
            mask: Possible dose mask tensor (B, 1, H, W, D)

        Returns:
            Scalar loss value
        """
        # Apply mask to both predictions and targets
        masked_pred = pred * mask
        masked_target = target * mask

        # Compute MSE only on masked regions
        squared_diff = (masked_pred - masked_target) ** 2
        loss = squared_diff.sum() / (mask.sum() + 1e-8)

        return loss


class MaskedHuberLoss(nn.Module):
    """
    Masked Huber Loss (Smooth L1 Loss).

    Combines the benefits of MAE and MSE - behaves like MSE for small errors
    and like MAE for large errors, making it more robust to outliers.
    """

    def __init__(self, delta: float = 1.0):
        """
        Args:
            delta: Threshold at which to switch from MSE to MAE behavior
        """
        super().__init__()
        self.delta = delta

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute masked Huber loss.

        Args:
            pred: Predicted dose tensor (B, 1, H, W, D)
            target: Target dose tensor (B, 1, H, W, D)
            mask: Possible dose mask tensor (B, 1, H, W, D)

        Returns:
            Scalar loss value
        """
        # Apply mask to both predictions and targets
        masked_pred = pred * mask
        masked_target = target * mask

        # Compute absolute difference
        abs_diff = torch.abs(masked_pred - masked_target)

        # Huber loss formula
        quadratic = torch.clamp(abs_diff, max=self.delta)
        linear = abs_diff - quadratic
        huber = 0.5 * quadratic**2 + self.delta * linear

        loss = huber.sum() / (mask.sum() + 1e-8)

        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss function with weighted MAE and MSE components.

    This can help balance between accurate overall dose prediction (MSE)
    and robust handling of high-dose regions (MAE).
    """

    def __init__(self, mae_weight: float = 0.5, mse_weight: float = 0.5):
        """
        Args:
            mae_weight: Weight for MAE component
            mse_weight: Weight for MSE component
        """
        super().__init__()
        self.mae_weight = mae_weight
        self.mse_weight = mse_weight
        self.mae_loss = MaskedMAELoss()
        self.mse_loss = MaskedMSELoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute combined masked loss.

        Args:
            pred: Predicted dose tensor (B, 1, H, W, D)
            target: Target dose tensor (B, 1, H, W, D)
            mask: Possible dose mask tensor (B, 1, H, W, D)

        Returns:
            Scalar loss value
        """
        mae = self.mae_loss(pred, target, mask)
        mse = self.mse_loss(pred, target, mask)

        return self.mae_weight * mae + self.mse_weight * mse


def get_loss_function(loss_name: str = "mae", **kwargs) -> nn.Module:
    """
    Factory function to get a loss function by name.

    Args:
        loss_name: Name of the loss function ('mae', 'mse', 'huber', 'combined')
        **kwargs: Additional arguments to pass to the loss function

    Returns:
        Loss function module

    Raises:
        ValueError: If loss_name is not recognized
    """
    loss_functions = {
        "mae": MaskedMAELoss,
        "mse": MaskedMSELoss,
        "huber": MaskedHuberLoss,
        "combined": CombinedLoss,
    }

    if loss_name.lower() not in loss_functions:
        raise ValueError(
            f"Unknown loss function: {loss_name}. "
            f"Available options: {list(loss_functions.keys())}"
        )

    return loss_functions[loss_name.lower()](**kwargs)
