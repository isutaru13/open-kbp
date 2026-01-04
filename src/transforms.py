"""
MONAI transforms for OpenKBP data augmentation and preprocessing.
"""

from monai.transforms import (
    Compose,
    NormalizeIntensityd,
    RandFlipd,
    RandGaussianNoised,
    RandRotate90d,
    ToTensord,
)


def get_transforms(training: bool = True) -> Compose:
    """
    Get MONAI transforms for training or validation.

    Args:
        training: If True, include data augmentation transforms.
                  If False, only apply normalization and tensor conversion.

    Returns:
        Compose: A composed transform pipeline.
    """
    if training:
        return Compose(
            [
                # Normalize CT intensities (channel-wise normalization)
                NormalizeIntensityd(keys=["image"], channel_wise=True),
                # Data augmentation
                RandFlipd(keys=["image", "dose", "mask"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "dose", "mask"], prob=0.5, spatial_axis=1),
                RandRotate90d(
                    keys=["image", "dose", "mask"], prob=0.5, spatial_axes=(0, 1)
                ),
                RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.1),
                ToTensord(keys=["image", "dose", "mask"]),
            ]
        )
    else:
        return Compose(
            [
                NormalizeIntensityd(keys=["image"], channel_wise=True),
                ToTensord(keys=["image", "dose", "mask"]),
            ]
        )


def get_inference_transforms() -> Compose:
    """
    Get MONAI transforms for inference (no dose in data).

    Returns:
        Compose: A composed transform pipeline for inference.
    """
    return Compose(
        [
            NormalizeIntensityd(keys=["image"], channel_wise=True),
            ToTensord(keys=["image", "mask"]),
        ]
    )
