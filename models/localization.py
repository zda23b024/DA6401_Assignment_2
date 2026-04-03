"""Localization modules
"""

import torch
import torch.nn as nn

from models.vgg11 import VGG11Encoder
from models.layers import CustomDropout


class VGG11Localizer(nn.Module):
    """VGG11-based localizer."""

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11Localizer model.

        Args:
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the localization head.
        """
        super(VGG11Localizer, self).__init__()

        # Shared encoder (same as classification)
        self.encoder = VGG11Encoder(in_channels=in_channels)

        # Regression head
        self.regressor = nn.Sequential(
            nn.Flatten(),  # [B, 512, 7, 7] → [B, 25088]

            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),

            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),

            nn.Linear(512, 4)  # Output: [x_center, y_center, width, height]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for localization model.

        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Bounding box coordinates [B, 4]
        """
        # Extract deep features
        features = self.encoder(x)  # [B, 512, 7, 7]

        # Predict bounding boxes
        bbox = self.regressor(features)  # [B, 4]

        return bbox