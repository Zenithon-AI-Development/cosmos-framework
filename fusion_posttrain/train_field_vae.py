# SPDX-License-Identifier: OpenMDW-1.1
"""Stage-1 field-VAE training (DDP, bf16, grad-checkpointed).

Trains FieldVAE on a subset of GORGON DD_IFE trajectories at full spatial
resolution (theta=256, r=2400), 16-frame temporal windows, batch=1/GPU.

Loss (all in NORMALIZED space -- equalizes per-channel scale per SESSION 5.1):
  - per-channel weighted L1 recon
  - gradient/edge term: L1 of finite differences along r (W) and theta (H),
    protecting the steep shock/ablation fronts that 16x compression risks.
  - small KL (default 1e-6) -- this is a tokenizer, not a generative prior.

Run (single node, 4 GPUs):
  torchrun --nproc_per_node=4 train_field_vae.py \
      --train_dir DATA/train --valid_dir DATA/valid \
      --ids_file refs/subset_train50.txt --valid_ids_file refs/valid8.txt \
      --epochs 40 --out_dir outputs
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.field_dataset import FieldClipDataset, list_ids  # noqa: E402
from model.field_vae import FieldVAE  # noqa: E402
from norm import CHANNELS  # noqa: E402


def is_dist():
    return dist.is_available() and dist.is_initialized()


def rank0():
    return (not is_dist()) or dist.get_rank() == 0


def setup_ddp():
    if "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group("nccl")
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        return local
    return 0


def read_ids(path, data_dir):
    if path and os.path.exists(path):
        with open(path) as f:
            return [ln.strip() for ln in f if ln.strip()]
    return list_ids(data_dir)


def spatial_grad_l1(recon, x):
    """L1 error of finite differences along H(theta) and W(r). Inputs [B,C,T,H,W];
    returns per-channel [C]."""
    dr_r = recon[..., 1:] - recon[..., :-1]
    dr_x = x[..., 1:] - x[..., :-1]
    dh_r = recon[..., 1:, :] - recon[..., :-1, :]
    dh_x = x[..., 1:, :] - x[..., :-1, :]
    er = (dr_r - dr_x).abs().mean(dim=(0, 2, 3, 4))
    eh = (dh_r - dh_x).abs().mean(dim=(0, 2, 3, 4))
    return er + eh  # [C]


def per_channel_l1(recon, x):
    return (recon - x).abs().mean(dim=(0, 2, 3, 4))  # [C]


@torch.no_grad()
def evaluate(model, loader, device, cw):
    model.eval()
    tot = torch.zeros(len(CHANNELS), device=device)
    totg = torch.zeros(len(CHANNELS), device=device)
    n = 0
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            recon, mu, logvar, kl = model(x)
        recon = recon.float()
        tot += per_channel_l1(recon, x)
        totg += spatial_grad_l1(recon, x)
        n += 1
    if is_dist():
        dist.all_reduce(tot); dist.all_reduce(totg)
        cnt = torch.tensor([n], device=device); dist.all_reduce(cnt); n = cnt.item()
    return (tot / max(n, 1)).cpu(), (totg / max(n, 1)).cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--valid_dir", required=True)
    ap.add_argument("--ids_file", default=None)
    ap.add_argument("--valid_ids_file", default=None)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--kl_weight", type=float, default=1e-6)
    ap.add_argument("--grad_weight", type=float, default=1.0)
    ap.add_argument("--channel_weights", default=None,
                    help="comma-separated 7 floats; default uniform")
    ap.add_argument("--batch_size", type=int, default=1, help="clips per GPU")
    ap.add_argument("--no_checkpoint", action="store_true",
                    help="disable grad checkpointing (faster; needs more mem -- OK on A100-80GB)")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--wandb", default=None, help="W&B project name; enables logging if set")
    ap.add_argument("--wandb_run", default=None, help="W&B run name")
    ap.add_argument("--wandb_id", default=None,
                    help="fixed W&B run id -> resumes same run after spot preemption")
    ap.add_argument("--ckpt_artifact_every", type=int, default=10,
                    help="log a W&B model artifact every N epochs (plus final)")
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=0, help=">0 caps steps (smoke test)")
    ap.add_argument("--out_dir", default="outputs")
    args = ap.parse_args()

    local = setup_ddp()
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    world = dist.get_world_size() if is_dist() else 1

    if args.channel_weights:
        cw = torch.tensor([float(v) for v in args.channel_weights.split(",")], dtype=torch.float32)
        assert cw.numel() == len(CHANNELS)
    else:
        cw = torch.ones(len(CHANNELS), dtype=torch.float32)
    cw = cw.to(device).view(1, -1, 1, 1, 1)

    train_ids = read_ids(args.ids_file, args.train_dir)
    valid_ids = read_ids(args.valid_ids_file, args.valid_dir)
    if rank0():
        print(f"world={world} train_ids={len(train_ids)} valid_ids={len(valid_ids)}", flush=True)

    train_ds = FieldClipDataset(args.train_dir, train_ids, window=args.window)
    valid_ds = FieldClipDataset(args.valid_dir, valid_ids, window=args.window)
    if rank0():
        print(f"train clips={len(train_ds)} valid clips={len(valid_ds)}", flush=True)

    if is_dist():
        tr_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)
        va_sampler = DistributedSampler(valid_ds, shuffle=False, drop_last=False)
    else:
        tr_sampler = va_sampler = None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=tr_sampler,
                              shuffle=(tr_sampler is None), num_workers=args.num_workers,
                              pin_memory=True, drop_last=True, persistent_workers=args.num_workers > 0)
    valid_loader = DataLoader(valid_ds, batch_size=1, sampler=va_sampler,
                              shuffle=False, num_workers=2, pin_memory=True)

    model = FieldVAE(use_checkpoint=not args.no_checkpoint).to(device)
    if rank0():
        print(f"params={sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)
    if is_dist():
        model = DDP(model, device_ids=[local], find_unused_parameters=False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.95))
    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "train_log.jsonl")
    ckpt_path = os.path.join(args.out_dir, "field_vae_last.pt")

    wb = None
    if args.wandb and rank0():
        try:
            import wandb as wb
            # fixed id + resume="allow" => a preempted run CONTINUES the same
            # W&B run instead of spawning a new one on every recovery.
            wb.init(project=args.wandb, name=args.wandb_run, id=args.wandb_id,
                    resume="allow", config=vars(args))
            wb.summary["params_M"] = sum(p.numel() for p in model.parameters()) / 1e6
        except Exception as e:
            print(f"[wandb] disabled ({e})", flush=True); wb = None

    def save_ckpt(epoch, gstep, artifact=False):
        """Atomic local/GCS-mount write (tmp + os.replace) so a spot preemption
        mid-write can never corrupt the resume file. Carries optimizer state +
        epoch + gstep so a recovered job continues exactly where it stopped.
        When artifact=True, ALSO logs the ckpt as a versioned W&B model artifact
        (portable backup / Stage-3 handoff); resume can pull it if GCS is empty."""
        sd = (model.module if is_dist() else model).state_dict()
        tmp = ckpt_path + ".tmp"
        torch.save({"model": sd, "opt": opt.state_dict(), "epoch": epoch,
                    "gstep": gstep, "args": vars(args)}, tmp)
        os.replace(tmp, ckpt_path)
        if artifact and wb is not None:
            try:
                art = wb.Artifact("field_vae", type="model",
                                  metadata={"epoch": epoch, "gstep": gstep})
                art.add_file(ckpt_path)
                wb.log_artifact(art, aliases=["latest", f"epoch-{epoch}"])
            except Exception as e:
                print(f"[wandb] artifact upload failed ({e})", flush=True)

    # --- RESUME (spot-preemption recovery) -------------------------------
    # PRIMARY store = the SkyPilot-mounted gs://zen_ml_checkpoints bucket, so the
    # checkpoint survives instance loss and resume is a fast local read. If the
    # mount is empty (e.g. mount failed) we FALL BACK to pulling the latest W&B
    # model artifact and resuming from that. rank0 stages the file onto the shared
    # mount; a barrier then lets every rank load the same local file.
    start_epoch, gstep = 0, 0
    if rank0() and not os.path.exists(ckpt_path) and wb is not None:
        try:
            art = wb.use_artifact(f"{args.wandb}/field_vae:latest", type="model")
            d = art.download()
            src = os.path.join(d, os.path.basename(ckpt_path))
            if os.path.exists(src):
                shutil.copy(src, ckpt_path)
                print(f"RESUME: pulled latest W&B artifact -> {ckpt_path}", flush=True)
        except Exception as e:
            print(f"[wandb] no resumable artifact ({e})", flush=True)
    if is_dist():
        dist.barrier()  # ensure the staged ckpt is visible to all ranks
    resumed = False
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        (model.module if is_dist() else model).load_state_dict(ck["model"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        ep = ck.get("epoch", -1)
        start_epoch = (ep + 1) if isinstance(ep, int) else 0
        gstep = int(ck.get("gstep", 0))
        resumed = True
        if rank0():
            print(f"RESUME from {ckpt_path}: start_epoch={start_epoch} gstep={gstep}", flush=True)
    if rank0() and not resumed:
        open(log_path, "w").close()

    def do_eval_log(epoch, tag):
        l1c, gradc = evaluate(model, valid_loader, device, cw)
        model.train()
        if rank0():
            rec = {"epoch": epoch, "gstep": gstep,
                   "val_l1": {c: round(l1c[i].item(), 5) for i, c in enumerate(CHANNELS)},
                   "val_grad_l1": {c: round(gradc[i].item(), 5) for i, c in enumerate(CHANNELS)}}
            print(f"VAL[{tag}]", json.dumps(rec["val_l1"]), flush=True)
            if wb:
                wb.log({**{f"val_l1/{c}": l1c[i].item() for i, c in enumerate(CHANNELS)},
                        **{f"val_grad/{c}": gradc[i].item() for i, c in enumerate(CHANNELS)}},
                       step=gstep)
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        return l1c, gradc

    t0 = time.time()
    steps_this_run = 0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        if tr_sampler is not None:
            tr_sampler.set_epoch(epoch)
        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                recon, mu, logvar, kl = model(x)
            recon = recon.float()
            recon_loss = ((recon - x).abs() * cw).mean()
            grad_loss = spatial_grad_l1(recon, x).mean()
            loss = recon_loss + args.grad_weight * grad_loss + args.kl_weight * kl.float()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            gstep += 1
            steps_this_run += 1
            if rank0() and gstep % 10 == 0:
                print(f"e{epoch} step{gstep} loss={loss.item():.4f} "
                      f"recon={recon_loss.item():.4f} grad={grad_loss.item():.4f} "
                      f"kl={kl.item():.4f} {(time.time()-t0)/max(steps_this_run,1):.2f}s/it", flush=True)
                if wb:
                    wb.log({"train/loss": loss.item(), "train/recon": recon_loss.item(),
                            "train/grad": grad_loss.item(), "train/kl": kl.item(),
                            "epoch": epoch}, step=gstep)
            if args.max_steps and gstep >= args.max_steps:
                break

        # checkpoint EVERY epoch (persisted to the mounted bucket) so a spot
        # preemption loses at most one epoch of work.
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            do_eval_log(epoch, "epoch")
        if rank0():
            save_ckpt(epoch, gstep,
                      artifact=((epoch + 1) % args.ckpt_artifact_every == 0))
        if is_dist():
            dist.barrier()  # keep all ranks aligned before next epoch / exit
        if args.max_steps and gstep >= args.max_steps:
            break

    # final eval + ckpt + W&B artifact (also covers a max_steps smoke exit).
    l1c, gradc = do_eval_log("final", "final")
    if rank0():
        save_ckpt(args.epochs - 1, gstep, artifact=True)  # final portable artifact
        if wb:
            wb.finish()
        print("DONE", flush=True)
    if is_dist():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
