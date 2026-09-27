"""ResNet18 Spatial Feature Extractor Module for STORMFUSION (Step 8A).

Extracts spatial feature representations from 3-channel satellite imagery frames
(TIR1, WV, TIR2) using torchvision ResNet18 without pretrained weights (pretrained: false).

Classification heads are removed, outputting 512-channel spatial feature maps [B, 512, H_out, W_out].
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18SpatialEncoder(nn.Module):
    """ResNet18 spatial feature extractor module."""

    def __init__(self, in_channels: int = 3, pretrained: bool = False):
        super(ResNet18SpatialEncoder, self).__init__()

        self.in_channels = in_channels
        self.pretrained = pretrained

        # Load ResNet18 backbone without pretrained weights
        if pretrained:
            raise ValueError(
                "Pretrained weights are disabled for STORMFUSION Step 8A. "
                "Real satellite data is not yet available and ImageNet pretraining is not appropriate "
                "without a deliberate transfer-learning experiment."
            )

        weights = None
        backbone = models.resnet18(weights=weights)

        # Modify first convolution if in_channels != 3
        if in_channels != 3:
            orig_conv = backbone.conv1
            backbone.conv1 = nn.Conv2d(
                in_channels,
                orig_conv.out_channels,
                kernel_size=orig_conv.kernel_size,
                stride=orig_conv.stride,
                padding=orig_conv.padding,
                bias=orig_conv.bias is not None,
            )

        # Extract spatial feature representation blocks (removing avgpool and fc)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.out_channels = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for a 2D batch of image frames.

        Args:
            x: Image tensor [B, C_in, H, W]

        Returns:
            Spatial feature map [B, 512, H_out, W_out]
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x
