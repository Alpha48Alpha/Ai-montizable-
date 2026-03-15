"""Convolutional encoder for pixel / grid-image observations.

CNNEncoder can be used as a drop-in observation backbone when an
environment exposes 2-D image observations (H × W × C layout) instead
of a flat vector.

Typical usage
-------------
::

    encoder = CNNEncoder(in_channels=3, img_size=64, out_dim=256)
    # Attach an MLP head on top:
    policy_head = MLP(256, act_dim, [128])
    # In the forward pass:
    features = encoder(pixel_obs)       # (B, 256)
    logits   = policy_head(features)    # (B, act_dim)

Extension hooks
---------------
- Replace Conv layers with a ViT patch-embedding for large images
- Add a recurrent layer (LSTM) for partial observability
- Expose ``latent_dim`` for world-model agents that predict future latents
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """Small convolutional feature extractor.

    Parameters
    ----------
    in_channels : int
        Number of input channels (e.g. 3 for RGB, 1 for greyscale).
    img_size : int
        Assumed square spatial resolution (pixels).  The encoder is
        constructed so the spatial dimensions collapse to 1×1 before
        the linear projection.
    out_dim : int
        Dimensionality of the output feature vector.
    """

    def __init__(
        self,
        in_channels: int = 3,
        img_size: int = 64,
        out_dim: int = 256,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.img_size = img_size
        self.out_dim = out_dim

        # Three conv blocks: each halves spatial size
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        # Compute flat size after conv
        flat_size = self._compute_flat_size(in_channels, img_size)
        self.fc = nn.Linear(flat_size, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images.

        Parameters
        ----------
        x : torch.Tensor  shape ``(B, C, H, W)`` or ``(B, H, W, C)``
            Images normalised to [0, 1].

        Returns
        -------
        features : torch.Tensor  shape ``(B, out_dim)``
        """
        if x.dim() == 4 and x.shape[-1] == self.in_channels:
            # (B, H, W, C) → (B, C, H, W)
            x = x.permute(0, 3, 1, 2)
        h = self.conv(x)
        h = h.flatten(start_dim=1)
        return self.fc(h)

    # ------------------------------------------------------------------ #

    def _compute_flat_size(self, in_channels: int, img_size: int) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, img_size, img_size)
            out = self.conv(dummy)
        return int(math.prod(out.shape[1:]))
