# SPDX-License-Identifier: OpenMDW-1.1
"""Stage-1 field-VAE reconstruction eval -- the go/no-go evidence.

For each held-out valid trajectory we pick TWO windows (SESSION decision D4):
  (a) a LASER-ON window  -- the window whose frames contain the laser_power argmax
  (b) a LATE FINE-dt window -- the final window (fast stagnation/burn fronts)
and report, in PHYSICAL units (denormalized), per channel SEPARATELY:
  - relative L1 = mean|recon-x| / (mean|x|+eps)   and normalized-space L1
  - front/gradient: radial-gradient (d/dr) L1, front-location error in MICRONS
    (argmax|d Te/dr| radius via coords_r), and error on top-5% high-|grad| pixels
  - conservation: integrated-mass drift for rho_CH, rho_DT using polar element
    dV ~ r dr dtheta (coords_r, coords_theta)
PNG triptychs (input | recon | error) per channel per window -> outputs/recon/.
Metrics dumped to outputs/recon/metrics.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import zarr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from data.field_dataset import window_starts  # noqa: E402
from model.field_vae import FieldVAE  # noqa: E402
from norm import CHANNELS, FieldNorm  # noqa: E402

EPS = 1e-8


def load_model(ckpt, device):
    m = FieldVAE(use_checkpoint=False).to(device)
    sd = torch.load(ckpt, map_location=device)
    m.load_state_dict(sd["model"] if "model" in sd else sd)
    m.eval()
    return m


def pick_windows(g, window):
    n = int(g["fields"].shape[0])
    lp = np.asarray(g["laser_power"][:])
    laser_t = int(np.argmax(lp))
    starts = window_starts(n, window)
    # laser-on window = the one containing laser_t
    laser_start = min(starts, key=lambda s: 0 if s <= laser_t < s + window else abs(s - laser_t))
    late_start = starts[-1]
    return [("laser", laser_start), ("late", late_start)], n


def polar_area(coords_r, coords_theta):
    """dV element ~ r dr dtheta on the (theta=H, r=W) grid -> [H, W] weights."""
    r = np.asarray(coords_r, dtype=np.float64)            # [W]
    th = np.asarray(coords_theta, dtype=np.float64)       # [H]
    dr = np.gradient(r)                                    # [W]
    dth = np.gradient(th)                                  # [H]
    return (dth[:, None] * (r * dr)[None, :])              # [H, W]


@torch.no_grad()
def recon_window(model, x_norm, device):
    x = x_norm.unsqueeze(0).to(device)  # [1,C,T,H,W]
    if device.type == "cuda":
        with torch.autocast("cuda", dtype=torch.bfloat16):
            recon, *_ = model(x)
        recon = recon.float()
    else:
        recon, *_ = model(x)
    return recon.squeeze(0).cpu()  # [C,T,H,W]


def front_radius_microns(field_2d, coords_r):
    """argmax|d field / dr| radius -> microns. field_2d [H,W] (theta,r)."""
    grad = np.abs(np.diff(field_2d, axis=1))      # [H, W-1]
    prof = grad.mean(axis=0)                       # [W-1] mean over theta
    idx = int(np.argmax(prof))
    return float(coords_r[idx]) * 1e6              # m -> microns


def triptych(inp, rec, ch_name, tid, tag, t_phys, dt_phys, out_dir):
    err = rec - inp
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    vmin, vmax = float(np.percentile(inp, 1)), float(np.percentile(inp, 99))
    for ax, data, title in zip(axes, [inp, rec, err],
                               ["input", "recon", "error (rec-inp)"]):
        if title.startswith("error"):
            a = float(np.percentile(np.abs(err), 99)) + EPS
            im = ax.imshow(data, aspect="auto", cmap="RdBu_r", vmin=-a, vmax=a)
        else:
            im = ax.imshow(data, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"{ch_name} {title}")
        ax.set_xlabel("r (idx)"); ax.set_ylabel("theta (idx)")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"{tid} [{tag}]  t={t_phys:.3e}s  local dt={dt_phys:.2e}s")
    fig.tight_layout()
    p = os.path.join(out_dir, f"{tid}_{tag}_{ch_name}.png")
    fig.savefig(p, dpi=90); plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--valid_dir", required=True)
    ap.add_argument("--valid_ids_file", default=None)
    ap.add_argument("--n_traj", type=int, default=4)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--mid_frame", type=int, default=8, help="frame within window to plot")
    ap.add_argument("--out_dir", default="outputs/recon")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    model = load_model(args.ckpt, device)
    norm = FieldNorm()

    if args.valid_ids_file and os.path.exists(args.valid_ids_file):
        ids = [ln.strip() for ln in open(args.valid_ids_file) if ln.strip()]
    else:
        ids = sorted(d[:-5] for d in os.listdir(args.valid_dir) if d.endswith(".zarr"))
    ids = ids[: args.n_traj]

    results = {}
    for tid in ids:
        g = zarr.open_group(os.path.join(args.valid_dir, f"{tid}.zarr"), mode="r")
        root = g.attrs.asdict()
        coords_r = np.asarray(root.get("coords_r"))         # [W] meters
        coords_theta = np.asarray(root.get("coords_theta"))  # [H] rad
        area = polar_area(coords_r, coords_theta)            # [H,W]
        windows, n = pick_windows(g, args.window)
        results[tid] = {}
        for tag, t0 in windows:
            sl = slice(t0, t0 + args.window)
            raw = np.ascontiguousarray(g["fields"][sl])      # [T,H,W,C]
            time = np.asarray(g["time"][sl])
            x_phys = torch.from_numpy(raw).permute(3, 0, 1, 2).float()  # [C,T,H,W] physical
            x_norm = norm.normalize(x_phys, cdim=0)
            rec_norm = recon_window(model, x_norm, device)
            rec_phys = norm.denormalize(rec_norm, cdim=0)    # back to physical

            f = args.mid_frame if args.mid_frame < args.window else args.window // 2
            dt = float(time[f] - time[f - 1]) if f > 0 else float(time[1] - time[0])
            entry = {"t0": int(t0), "n_steps": int(n), "t_phys": float(time[f]),
                     "local_dt": dt, "channels": {}, "pngs": []}

            for ci, ch in enumerate(CHANNELS):
                xc = x_phys[ci].numpy()        # [T,H,W] physical
                rc = rec_phys[ci].numpy()
                xn = x_norm[ci].numpy()
                rn = rec_norm[ci].numpy()
                rel_l1 = float(np.mean(np.abs(rc - xc)) / (np.mean(np.abs(xc)) + EPS))
                norm_l1 = float(np.mean(np.abs(rn - xn)))
                # radial gradient error (physical)
                gx = np.abs(np.diff(xc, axis=-1)); gr = np.abs(np.diff(rc, axis=-1))
                grad_l1 = float(np.mean(np.abs(gr - gx)))
                # high-grad top-5% pixel error vs overall (normalized space)
                gmag = np.abs(np.diff(xn, axis=-1))
                thr = np.percentile(gmag, 95)
                mask = gmag >= thr
                err_n = np.abs(rn - xn)[..., 1:]
                hi_err = float(err_n[mask].mean()) if mask.any() else 0.0
                ov_err = float(err_n.mean())
                entry["channels"][ch] = {
                    "rel_l1": round(rel_l1, 5), "norm_l1": round(norm_l1, 5),
                    "grad_l1_phys": grad_l1, "hi_grad_err": round(hi_err, 5),
                    "overall_err": round(ov_err, 5),
                    "hi_grad_ratio": round(hi_err / (ov_err + EPS), 3),
                }
                # front-location (Te) in microns
                if ch == "Te":
                    fr_x = front_radius_microns(xc[f], coords_r)
                    fr_r = front_radius_microns(rc[f], coords_r)
                    entry["front_Te_microns"] = {"input": round(fr_x, 3),
                                                 "recon": round(fr_r, 3),
                                                 "abs_err_um": round(abs(fr_r - fr_x), 3)}
                # conservation for densities: integrated mass drift
                if ch in ("rho_CH", "rho_DT"):
                    m_x = float(np.sum(xc[f] * area)); m_r = float(np.sum(rc[f] * area))
                    entry["channels"][ch]["mass_drift_frac"] = round(
                        (m_r - m_x) / (abs(m_x) + EPS), 5)
                # triptych at mid frame
                p = triptych(xc[f], rc[f], ch, tid, tag, float(time[f]), dt, args.out_dir)
                entry["pngs"].append(os.path.basename(p))
            results[tid][tag] = entry
            print(f"{tid} [{tag}] t={entry['t_phys']:.3e}s dt={dt:.2e}s "
                  f"Te rel_l1={entry['channels']['Te']['rel_l1']} "
                  f"front_err_um={entry.get('front_Te_microns',{}).get('abs_err_um')}", flush=True)

    out = os.path.join(args.out_dir, "metrics.json")
    with open(out, "w") as fp:
        json.dump(results, fp, indent=2)
    print("WROTE", out, flush=True)


if __name__ == "__main__":
    main()
