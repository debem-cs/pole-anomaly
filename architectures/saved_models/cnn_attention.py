"""Optimized Hybrid CNN + multi-head self-attention for time-series classification.

Adapted from the original CNN+Attention architecture with the following
optimizations for the anomaly classification task:

* Reduced convolutional stem dimensions from (32, 64, 64) to (16, 32, 64)
  to cut parameters in the heaviest part of the network.
* GroupNorm(8) replaces BatchNorm: no running statistics → train/eval
  behaviour is identical and the stem is amplitude-invariant at inference.
* 4 attention heads and 2 encoder layers on a 64-dim embedding.

Design rationale:
* The CNN front-end extracts local shape primitives (peaks, ramps, plateaus).
* The Transformer encoder block contextualizes those local features globally
  via multi-head self-attention, allowing the model to discriminate a true
  radiological event from a BBAG artefact based on long-range context.
* A learnable [CLS] token aggregates the attended sequence into a single
  vector that feeds the linear classifier (à la BERT).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class _ConvStem(nn.Module):
    """Three Conv-GN-ReLU blocks reducing temporal length by ~8 (3x stride 2).

    GroupNorm(8) replaces BatchNorm: no running statistics → train/eval
    behaviour is identical and the stem is amplitude-invariant at inference.
    """

    def __init__(self, in_ch: int, dims=(16, 32, 64)):
        super().__init__()
        d1, d2, d3 = dims
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, d1, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(8, d1), nn.ReLU(inplace=True),
            nn.Conv1d(d1, d2, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GroupNorm(8, d2), nn.ReLU(inplace=True),
            nn.Conv1d(d2, d3, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, d3), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)  # (B, d3, T/8)


class _SinusoidalPosEnc(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani 2017)."""

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        return x + self.pe[:, : x.size(1)]


class CNNAttention(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        conv_dims=(16, 32, 64),
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.stem = _ConvStem(in_channels, dims=conv_dims)
        # Project last conv channels -> d_model if needed
        self.proj = (nn.Conv1d(conv_dims[-1], d_model, 1)
                     if conv_dims[-1] != d_model else nn.Identity())

        self.pos_enc = _SinusoidalPosEnc(d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

        # For Grad-CAM hooks: target the last conv layer in the stem.
        self.last_conv_features = self.stem.net[-3]  # last Conv1d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T)
        feats = self.stem(x)              # (B, C, T')
        feats = self.proj(feats)          # (B, d_model, T')
        feats = feats.transpose(1, 2)     # (B, T', d_model)
        feats = self.pos_enc(feats)

        # Prepend CLS token
        B = feats.size(0)
        cls = self.cls_token.expand(B, -1, -1)   # (B, 1, d_model)
        seq = torch.cat([cls, feats], dim=1)     # (B, 1+T', d_model)

        seq = self.transformer(seq)
        seq = self.norm(seq)
        cls_out = seq[:, 0]                       # (B, d_model)
        return self.classifier(cls_out)
