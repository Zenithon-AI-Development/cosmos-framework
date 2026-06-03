# SESSION 01 — HANDOFF (resume point)

> Read this WITH `SESSION_01_field_vae_pilot.md` (the original spec). This file = current state +
> verified facts + decisions + exactly what's left. Working dir: `fusion_posttrain/`
> (repo root `/home/abetharana/cosmos-framework`). Resuming agent has zero prior context.

## STATUS: ~40% done. Data pipeline + model BUILT & verified. Train/eval/sky/launch REMAIN.

---
## A. §8 ENVIRONMENT — all checked, all GREEN (do NOT re-derive)
- **SkyPilot** 0.11.1; `sky check gcp` → GCP enabled (compute+storage).
- **Identity**: gcloud active = service account `skypilot-v1@dataset-gen-1-473109.iam.gserviceaccount.com`,
  project `dataset-gen-1-473109`. GCS read to `gs://zen_ml_data/...` CONFIRMED (834 train zarr listed).
  Same SA is used by SkyPilot workers → creds consistent local↔cloud.
- **Deps local**: torch 2.10.0+cu128, zarr 3.1.5, gcsfs 2026.2.0 all import.
- **NO LOCAL GPU** (`torch.cuda.is_available()==False`, nvidia-smi fails). → build/smoke-test on CPU,
  TRAIN on SkyPilot. `matplotlib` NOT yet verified (eval needs it; add to sky setup + smoke check).

## B. DATA FACTS (verified against the bucket, supersede the doc where they differ)
- Bucket: `gs://zen_ml_data/GORGON/DD_IFE_native/`  data/{train=834, valid=102, test=105}/<id>.zarr (zarr v3).
- `fields`: `[N,256,2400,7]` float32, layout **THWC**, chunks `(1,256,2400,7)` (per-frame → window slicing cheap).
- **N (timesteps) VARIES**: xy13632=88, xy13634=124, xy13635=121. The doc's "88" is one example only.
  Our dataset reads N per-store, so it's handled.
- Channel order: `["Te","Ti","Tr","rho_CH","rho_DT","vvec_0","vvec_2"]`.
- `coords_r` (2400): 2.5e-7 .. 1.2e-3 m, monotonic↑. `coords_theta` (256): 0 .. π. In root zarr attrs
  (also: n_r, n_theta, boundary_masks, geometry, source_trajectory).
- `laser_power[N]`, `time[N]`. Laser argmax ≈ t-idx 24–34; `lp>1%` spans nearly whole trajectory.
- **dt IS NON-UNIFORM (USER-FLAGGED, important)**: ~5e-11 s early → ~5e-12 s late (10–27× refine into the
  stagnation/burn phase). A fixed 16-frame window ≈ 800 ps early but ≈ 80 ps late. The fast-moving fronts
  (the hard physics) live in the LATE fine-dt phase.

## C. NORMALIZATION — PINNED & VERIFIED (use exactly this; in `norm.py`)
stats.yaml `*_arcsinh` family was computed as **plain `arcsinh(x)`** (no inner scale). Verified:
arcsinh(max_Te 5981)=9.39=max_arcsinh[Te]; arcsinh(min_vvec_0 -4.075e6)=-15.91=min_arcsinh[vvec_0].
```
normalize:   y = arcsinh(x);  x' = (y - mean_arcsinh) / std_arcsinh     (per channel)
denormalize: x = sinh(x' * std_arcsinh + mean_arcsinh)
```
Tested on REAL valid frames: no NaNs, all 7 channels sane. `Tr` is low-variance (per-clip std ~0.22) —
confirms §5.1: keep loss in normalized space (equalizes channel scale), expose per-channel weights, note in results.
`refs/stats.yaml` already downloaded locally.

## D. KEY DECISIONS (locked)
1. **Model = clean Wan-STYLE 3D ResBlock VAE, NOT the vendored streaming Wan class.** Rationale: the only
   Stage-3 requirement is the latent INTERFACE (16ch, 16× spatial, 4× temporal, patch_spatial=2 compatible),
   not Wan's block identity; and Wan's full-forward (non-cache) temporal path is ambiguous (temporal conv only
   runs on feat_cache). Geometry reproduced & SHAPE-VERIFIED.
2. **Temporal compression in native frame-index space** (matches DiT fixed 4× factor). Do NOT resample to
   uniform time now (would blur fast late-time fronts). Non-uniform dt + variable N → document as **Stage-3
   data-pipeline flags** (uniform-time resample vs dt-conditioning) in RESULTS_01.
3. **Windowing default** (SESSION §5): full θ=256, full r=2400, 16-frame windows, batch=1/GPU, grad-checkpoint.
   If OOM: drop temporal window before cutting r. `window_starts` always includes the FINAL window.
4. **Eval must report BOTH a laser-on window AND a late/fine-dt window**, panels annotated with physical t(s)+local dt.
5. **Infra**: 4×A100-80GB **spot** via SkyPilot (NOT H100). ~6h wall, ~$30–60. Stop early if clearly pass/fail.

## E. FILES WRITTEN & VERIFIED
- `norm.py` — FieldNorm (normalize/denormalize), CHANNELS, load_stats. ✔ round-trip + real-data tested.
- `model/field_vae.py` — `FieldVAE(in_channels=7, z_dim=16, base=128, ch_mult=(1,2,4,4), nblocks=2, patch=2,
  use_checkpoint=True)`. space_to_depth/depth_to_space patch=2 (7→28→7); encoder 8×spatial/4×temporal +
  patch 2× = net 16×/4×; VAE reparam + KL. **183.2M params**. Self-test passes:
  `python model/field_vae.py` → latent (1,16,4,4,6) from (1,7,16,64,96), recon shape OK.
  (183M > doc's 100–150M sketch; KEPT deliberately so an under-capacity VAE can't *falsely* fail 16×; fits A100-80GB.)
- `data/field_dataset.py` — `FieldClipDataset(data_dir, ids, window=16)` returns
  `{x:[7,16,256,2400] normalized, id, t0, time[16], laser_power[16]}`. `list_ids`, `window_starts` helpers.
  Handles variable N. Per-worker lazy zarr group cache.
- `refs/stats.yaml`, dirs `data/ model/ outputs/recon/ refs/`.

## F. WHAT'S LEFT (tasks 3–6)  — build in this order, smoke-test on CPU before launching
1. **`train_field_vae.py`** (task 3): torchrun/DDP, bf16 autocast, model `use_checkpoint=True`, AdamW.
   Loss = per-channel weighted L1 (normalized space) + gradient/edge term (finite-diff along r=W and θ=H,
   protects shock fronts) + small KL (~1e-6). Default channel_weights uniform (expose `--channel_weights`).
   batch=1/GPU, window=16, full res. Log per-channel L1 each eval interval. Save ckpt to `outputs/`.
   Params to expose: --train_dir --valid_dir --ids_file --epochs --lr --kl_weight --grad_weight --out_dir.
2. **`eval_recon.py`** (task 4): load ckpt + valid set; for a few trajs pick (a) laser-on window (containing
   laser argmax) and (b) final fine-dt window. Denormalize (physical units) and compute:
   - per-channel relative L1 = mean|recon-x|/(mean|x|+eps) AND normalized-space L1 (report each channel
     SEPARATELY — never average into one number);
   - **front/gradient**: radial-gradient (dTe/dr) error + front-location error (argmax|∇| radius, → microns via
     coords_r); error on top-5% high-|∇| pixels vs overall;
   - **conservation**: mass drift ∫rho·r·dr·dθ for rho_CH & rho_DT (polar element ∝ r dr dθ via coords_r/theta);
   - PNG triptychs (input | recon | error) per channel at both windows, annotated t(s)+dt → `outputs/recon/`.
   Dump metrics JSON + print.
3. **subset manifest** (task 5): deterministic first 50 train ids + first 8 valid ids → e.g. `refs/subset_train50.txt`,
   `refs/valid8.txt` (use `gsutil ls .../data/train/ | sort | head -50`).
4. **`sky_field_vae.yaml`** (task 5): `resources:{accelerators: A100-80GB:4, use_spot:true}`; setup installs
   torch+zarr+gcsfs+pyyaml+matplotlib (uv/pip); run: `gsutil -m cp -r` the 50 train + 8 valid subset to local
   (~49GB, fine) then `torchrun --nproc_per_node=4 train_field_vae.py ...`. Use `sky jobs launch` (spot auto-recover).
   Worker GCS auth = same SA (already works).
5. **CPU smoke test** (task 5): 1-step train on a tiny synthetic/real-1-clip + 1 eval call before cloud launch.
6. **Launch + RESULTS_01.md** (task 6): `sky jobs launch`, monitor; then eval; write `RESULTS_01.md` with
   per-channel numbers, front/laser/conservation verdicts, the dt/variable-N Stage-3 flags, windowing choice,
   recon PNG refs, and a **GO (16× holds) / GO-AT-8× / NO-GO** call. DO NOT touch the DiT (that's Stage 3).

## G. GOTCHAS
- No Write/Edit tool in this agent set → author files via Bash heredocs (`cat > f <<'EOF'`).
- Importing `cosmos_framework.*` FAILS locally (missing loguru etc.) — that's fine, our code is self-contained.
- `caveman` SessionStart hook active (terse prose); write code/commits/docs normally.
- Task list (TaskCreate/Update) IDs: 1✔ 2✔ done; 3,4,5,6 pending.
