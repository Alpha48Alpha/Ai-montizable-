"""Neural network architectures.

MLP
---
Generic fully-connected network usable as a Q-network, policy head,
or value head.  Activation is ReLU between hidden layers (configurable).

PolicyMLP
---------
Wrapper that signals "this is a policy head"; identical architecture to
MLP but semantically distinguished for clarity.

ValueMLP
--------
MLP with output dimension fixed to 1 (scalar value function).

Extension hooks
---------------
- Replace the ReLU activations with LayerNorm + GELU for transformer-style
  feature extraction.
- Stack an LSTM / GRU on top of these MLPs for POMDPs / memory tasks.
- Expose a ``forward_with_features()`` method for world-model agents that
  need the penultimate-layer representation.
"""

from __future__ import annotations

from typing import Type

import torch
import torch.nn as nn


def _build_mlp(
    in_dim: int,
    out_dim: int,
    hidden_sizes: list[int],
    activation: Type[nn.Module] = nn.ReLU,
) -> nn.Sequential:
    """Construct a fully-connected network."""
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden_sizes:
        layers.extend([nn.Linear(prev, h), activation()])
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class MLP(nn.Module):
    """Generic multi-layer perceptron.

    Parameters
    ----------
    in_dim : int
    out_dim : int
    hidden_sizes : list[int]
    activation : nn.Module subclass
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_sizes: list[int] | None = None,
        activation: Type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [128, 128]
        self.net = _build_mlp(in_dim, out_dim, hidden_sizes, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PolicyMLP(MLP):
    """Policy-head MLP (same architecture as MLP; semantically distinct)."""


class ValueMLP(nn.Module):
    """Scalar value-function MLP (output dimension = 1).

    Parameters
    ----------
    in_dim : int
    hidden_sizes : list[int]
    """

    def __init__(
        self,
        in_dim: int,
        hidden_sizes: list[int] | None = None,
        activation: Type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [128, 128]
        self.net = _build_mlp(in_dim, 1, hidden_sizes, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
