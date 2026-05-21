"""Optimized 1D Residual Network for time-series classification.

Adapted from the original ResNet1D architecture with the following
optimizations for the anomaly classification task:

* Reduced channel widths from (64, 128, 128) to (16, 32, 64) to drastically
  cut parameter count while retaining sufficient capacity.
* Replaced AdaptiveAvgPool1d(1) with AdaptiveAvgPool1d(4) to preserve 4
  temporal bins, allowing the classifier to see shape topography.
* Added a hidden dense layer (Linear → ReLU → Dropout) before the final
  output, mirroring the proven 1D-CNN classifier head.
* GroupNorm(8) instead of BatchNorm1d everywhere: GN has no running stats
  so train and eval normalise identically — eliminates val-loss spikes.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernels=(7, 5, 3)):
        super().__init__()
        layers = []
        prev = in_ch
        for k in kernels:
            layers += [
                nn.Conv1d(prev, out_ch, kernel_size=k, padding=k // 2, bias=False),
                nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
                nn.ReLU(inplace=True),
            ]
            prev = out_ch
        # Drop the trailing ReLU so we can sum with the residual first
        layers = layers[:-1]
        self.body = nn.Sequential(*layers)
        self.shortcut = (
            nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, bias=False),
                          nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch))
            if in_ch != out_ch
            else nn.Identity()
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.body(x) + self.shortcut(x))


class ResNet1D(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 1,
                 widths=(16, 32, 64), dropout: float = 0.3):
        super().__init__()
        c1, c2, c3 = widths
        # Larger first kernel (15) to capture broad local context.
        self.block1 = _ResBlock(in_channels, c1, kernels=(15, 9, 5))
        self.pool1 = nn.MaxPool1d(2)
        self.block2 = _ResBlock(c1, c2, kernels=(7, 5, 3))
        self.pool2 = nn.MaxPool1d(2)
        self.block3 = _ResBlock(c2, c3, kernels=(7, 5, 3))

        # Regional Pooling - preserves 4 temporal bins to maintain shape topography
        self.gap = nn.AdaptiveAvgPool1d(4)

        # Classifier with hidden layer and dropout (mirrors 1D-CNN head)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c3 * 4, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
        )

        # For Grad-CAM hooks
        self.last_conv_features = self.block3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.block3(x)
        x = self.gap(x)
        return self.classifier(x)
