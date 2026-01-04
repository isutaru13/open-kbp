"""
MONAI transforms for OpenKBP data augmentation and preprocessing.

Augmentation strategies for dose prediction:
- "none": No augmentation (recommended for dose prediction - preserves beam geometry)
- "intensity": Only intensity augmentation (noise, contrast) - safe for dose prediction
- "geometric": Flip and rotation (may break dose-anatomy relationship!)
- "full": All augmentations (geometric + intensity) - use with caution

NOTE: Geometric augmentation (flip/rotate) may hurt dose prediction because:
1. Dose depends on beam angles (not provided as input)
2. Flipping anatomy changes the beam-anatomy relationship
3. Model learns inconsistent input-output pairs
"""

from typing import Literal

from monai.transforms import (
    Compose,
    NormalizeIntensityd,
    RandAdjustContrastd,
    RandFlipd,
    RandGaussianNoised,
    RandRotate90d,
    RandShiftIntensityd,
    ToTensord,
)

# Augmentation type options
AugmentationType = Literal["none", "intensity", "geometric", "full"]


def get_transforms(
    training: bool = True,
    augment_type: AugmentationType = "intensity",
) -> Compose:
    """
    Get MONAI transforms for training or validation.

    Args:
        training: If True, include data augmentation transforms.
                  If False, only apply normalization and tensor conversion.
        augment_type: Type of augmentation to apply during training:
            - "none": No augmentation (safest for dose prediction)
            - "intensity": Only intensity augmentation (noise, contrast) - RECOMMENDED
            - "geometric": Only geometric augmentation (flip, rotate) - may hurt results!
            - "full": All augmentations (geometric + intensity) - use with caution

    Returns:
        Compose: A composed transform pipeline.

    Note:
        For dose prediction, geometric augmentation (flip/rotate) may be harmful
        because dose distributions depend on beam angles which aren't provided
        as input. When anatomy is flipped, the dose pattern should change too,
        but we can't model that without beam information.
    """
    # Base transforms (always applied)
    base_transforms = [
        NormalizeIntensityd(keys=["image"], channel_wise=True),
    ]

    # Intensity augmentation (safe for dose prediction)
    intensity_augments = [
        RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.1),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.2),
        RandAdjustContrastd(keys=["image"], gamma=(0.9, 1.1), prob=0.2),
    ]

    # Geometric augmentation (may break dose-anatomy relationship!)
    geometric_augments = [
        RandFlipd(keys=["image", "dose", "mask"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "dose", "mask"], prob=0.5, spatial_axis=1),
        RandRotate90d(keys=["image", "dose", "mask"], prob=0.5, spatial_axes=(0, 1)),
    ]

    # Final conversion
    final_transforms = [
        ToTensord(keys=["image", "dose", "mask"]),
    ]

    if training:
        if augment_type == "none":
            transforms = base_transforms + final_transforms
        elif augment_type == "intensity":
            transforms = base_transforms + intensity_augments + final_transforms
        elif augment_type == "geometric":
            transforms = base_transforms + geometric_augments + final_transforms
        elif augment_type == "full":
            transforms = (
                base_transforms
                + geometric_augments
                + intensity_augments
                + final_transforms
            )
        else:
            raise ValueError(f"Unknown augment_type: {augment_type}")

        return Compose(transforms)
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


# Convenience functions for common configurations
def get_no_augment_transforms(training: bool = True) -> Compose:
    """Get transforms with no augmentation."""
    return get_transforms(training=training, augment_type="none")


def get_intensity_augment_transforms(training: bool = True) -> Compose:
    """Get transforms with intensity-only augmentation (recommended)."""
    return get_transforms(training=training, augment_type="intensity")


def get_full_augment_transforms(training: bool = True) -> Compose:
    """Get transforms with full augmentation (may hurt dose prediction)."""
    return get_transforms(training=training, augment_type="full")
