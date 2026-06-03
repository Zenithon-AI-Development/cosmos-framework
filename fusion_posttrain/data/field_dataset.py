# SPDX-License-Identifier: OpenMDW-1.1
"""zarr v3 -> torch dataset for GORGON DD_IFE_native trajectories.

Each item is one temporal clip of all 7 fields at FULL spatial resolution
(theta=256, r=2400), normalized per-channel (see norm.py). Layout on disk is
THWC [N,256,2400,7] with per-frame chunks (1,256,2400,7); slicing a window in T
reads only that window. Full r and full theta kept (SESSION default: no naive
radial crop -- physics fronts live at specific moving radii).

NOTE: trajectory length N varies (88 / 121 / 124 / ...), NOT a fixed 88, and the
time axis is NON-UNIFORM (dt ~5e-11 s early -> ~5e-12 s late as the implosion
refines). We keep temporal compression in native frame-index space (matches the
DiT's fixed 4x temporal_compression_factor); dt-uniformity is a Stage-3 concern.
We carry `time` and `laser_power` per clip so eval can label panels physically.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from norm import FieldNorm  # noqa: E402


def list_ids(data_dir: str):
    return sorted(d[:-5] for d in os.listdir(data_dir) if d.endswith(".zarr"))


def window_starts(n_steps: int, window: int):
    """Sliding windows, stride=window, ALWAYS including the final frames (last
    window clamped to n_steps-window) so the fast fine-dt stagnation/burn phase
    at the end of the implosion is covered."""
    if n_steps <= window:
        return [0]
    starts = list(range(0, n_steps - window + 1, window))
    if starts[-1] != n_steps - window:
        starts.append(n_steps - window)
    return starts


class FieldClipDataset(Dataset):
    def __init__(self, data_dir: str, ids, window: int = 16, stats_path: str | None = None):
        self.data_dir = data_dir
        self.ids = list(ids)
        self.window = window
        self.norm = FieldNorm(stats_path) if stats_path else FieldNorm()
        self._groups = {}  # per-worker lazy cache
        self.index = []
        for tid in self.ids:
            n = int(self._group(tid)["fields"].shape[0])
            for t0 in window_starts(n, window):
                self.index.append((tid, t0))

    def _group(self, tid: str):
        g = self._groups.get(tid)
        if g is None:
            g = zarr.open_group(os.path.join(self.data_dir, f"{tid}.zarr"), mode="r")
            self._groups[tid] = g
        return g

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        tid, t0 = self.index[i]
        g = self._group(tid)
        sl = slice(t0, t0 + self.window)
        arr = g["fields"][sl]  # [T,H,W,C]
        x = torch.from_numpy(np.ascontiguousarray(arr)).permute(3, 0, 1, 2)  # [C,T,H,W]
        x = self.norm.normalize(x, cdim=0).float()
        time = torch.from_numpy(np.ascontiguousarray(g["time"][sl])).float()
        lp = torch.from_numpy(np.ascontiguousarray(g["laser_power"][sl])).float()
        return {"x": x, "id": tid, "t0": t0, "time": time, "laser_power": lp}
