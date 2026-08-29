"""LCR-Seg model factory."""

from .cosine_segmentation_head import CosineSegmentationHead
from .unet import UNet2D

__all__ = ["CosineSegmentationHead", "UNet2D"]
