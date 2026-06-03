# SPDX-License-Identifier: OpenMDW-1.1
"""Per-channel normalization for GORGON fields.

Reverse-engineered from gs://zen_ml_data/GORGON/DD_IFE_native/stats.yaml and
VERIFIED: the stats' *_arcsinh family was computed as plain arcsinh(x) (NO inner
scale). Check: arcsinh(max_Te=5981)=9.39=max_arcsinh[Te]; arcsinh(min_vvec_0=
-4.075e6)=-15.91=min_arcsinh[vvec_0]. So the transform that maps each physical
field to ~zero-mean/unit-var (handling sign + huge dynamic range) is:

    y  = arcsinh(x)                         # plain, per element
    x' = (y - mean_arcsinh) / std_arcsinh   # per channel

Inverse (for physics metrics in physical units):
    x  = sinh(x' * std_arcsinh + mean_arcsinh)
"""
from __future__ import annotations

import os

import torch
import yaml

CHANNELS = ["Te", "Ti", "Tr", "rho_CH", "rho_DT", "vvec_0", "vvec_2"]
DEFAULT_STATS = os.path.join(os.path.dirname(__file__), "refs", "stats.yaml")


def load_stats(path: str = DEFAULT_STATS):
    with open(path) as f:
        s = yaml.safe_load(f)
    mean = torch.tensor([float(s["mean_arcsinh"][c]) for c in CHANNELS], dtype=torch.float32)
    std = torch.tensor([float(s["std_arcsinh"][c]) for c in CHANNELS], dtype=torch.float32)
    return mean, std  # each [7]


class FieldNorm:
    """Apply / invert per-channel arcsinh standardization. Channel dim = 0 by
    default (data tensors are [C,T,H,W]); pass cdim=1 for batched [B,C,T,H,W]."""

    def __init__(self, path: str = DEFAULT_STATS):
        self.mean, self.std = load_stats(path)

    def _shape(self, t: torch.Tensor, cdim: int):
        view = [1] * t.dim()
        view[cdim] = -1
        return (self.mean.to(t.device).view(view), self.std.to(t.device).view(view))

    def normalize(self, x: torch.Tensor, cdim: int = 0) -> torch.Tensor:
        mean, std = self._shape(x, cdim)
        return (torch.asinh(x) - mean) / std

    def denormalize(self, x: torch.Tensor, cdim: int = 1) -> torch.Tensor:
        mean, std = self._shape(x, cdim)
        return torch.sinh(x * std + mean)
