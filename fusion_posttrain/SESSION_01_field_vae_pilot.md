# SESSION 01 — Field-VAE Pilot (GORGON DD_IFE → Cosmos3-Nano, Option C)

> Self-contained handoff. The executing agent starts with ZERO prior context. Read this top to bottom.
> Working dir for all new code/artifacts: `fusion_posttrain/` (repo root = `/home/abetharana/cosmos-framework`).

## 0. Mission in one paragraph

We are post-training **Cosmos3-Nano** (an 8B Qwen3-VL MoT video diffusion world model) on **nuclear-fusion
simulation data** (GORGON inertial-fusion-energy runs). The data is NOT RGB video — it is 7-channel physical
fields on a structured polar (r,θ) mesh. The chosen strategy is **Option C**: train a *custom field tokenizer*
(a VAE that maps the 7 physical fields → a 16-channel latent at 16× spatial / 4× temporal compression, matching
the DiT's `vae2llm`/`llm2vae` interface), then SFT the Nano DiT on those latents. **THIS SESSION is only Stage 1:
the de-risk pilot.** Train the field VAE on a ~50-trajectory subset and answer ONE question: *does the VAE
reconstruct the physics — shock fronts, ablation/laser-heating front, all 7 fields — well enough at 16×
compression?* That go/no-go gates the whole (expensive) rest of the project. Do NOT touch the DiT this session.

## 1. Why we're doing it this way (background, so you don't re-derive it)

- Cosmos3-Nano's generator runs on **Wan2.2 VAE latents**: `pixels[3,T,H,W] → VAE → latent[16, T/4, H/16, W/16]
  → patchify(p=2) → vae2llm: Linear(p²·16 → hidden) → MoT transformer`. Confirmed in
  `cosmos_framework/model/vfm/mot/cosmos3_vfm_network.py` (`latent_channel_size=16`,
  `self.vae2llm = nn.Linear(self.latent_patch_size**2 * 16, hidden_size)`).
- The frozen Wan VAE only eats 3-ch RGB. Our data has 7 physical fields with huge dynamic range. Colormapping
  7→3 (Option A) throws away physics. So we replace the tokenizer (Option C).
- **Cost reality (already analyzed):** the DiT step is *fine-tuning from a strong init* = cheap (1–3 days, 8×H100).
  The VAE is the only *from-scratch* + *full-resolution* piece, so it is the dominant cost AND the dominant risk.
  Hence: pilot the VAE first, on cheap hardware, before committing to the DiT SFT.
- Full Option-C budget estimate (for context only): ~5–12 days on 8×H100 end-to-end, ~1,000–2,300 H100-GPU-hrs.

## 2. The dataset (verified against the bucket)

- **Location:** `gs://zen_ml_data/GORGON/DD_IFE_native/`
  - `stats.yaml` — per-channel normalization stats: keys `iqr`, `iqr_arcsinh`, `iqr_log10`, `iqr_log1p`,
    `max`, `min`, (and presumably `mean`/`std`/`median` — verify). Use these; do NOT recompute from scratch.
  - `data/{train,valid,test}/<id>.zarr/` — **zarr v3** stores. Counts: **train 834 / valid 102 / test 105**.
- **One trajectory** (e.g. `data/train/xy13632.zarr/`), ~978 MiB zstd on disk:
  - `fields`: shape **`[88, 256, 2400, 7]`** float32, **layout THWC**.
    - T = 88 timesteps (`n_steps=88` in root `zarr.json` attrs)
    - H = 256 = θ (poloidal angle; `coords_theta` array of 256 values in attrs)
    - W = 2400 = r (radial)
    - C = 7 channels, ORDER (from `channel_names` attr): **`["Te","Ti","Tr","rho_CH","rho_DT","vvec_0","vvec_2"]`**
      = electron temp, ion temp, radiation temp, CH (ablator) density, DT (fuel) density, velocity comp 0, velocity comp 2.
  - `laser_power`: `[88]` float32 — drive power per timestep (THE forcing input; matters for Stage 3 conditioning,
    and for THIS session it's the physical reason temperature fronts appear — use it to locate laser-heating events
    when picking eval frames).
  - `time`: `[88]` float32 — physical timestamp per step.
  - Root `zarr.json` attrs also carry `coords_theta`, `coords_r` (the mesh geometry), `source_trajectory`, `layout`.
- **Dynamic range is extreme** (from stats.yaml `max`): Ti up to ~1.45e5, vvec_0 ~7e5, temps ~thousands, densities ~tens.
  Velocities are SIGNED. → Per-channel **arcsinh** normalization is the natural choice (handles sign + huge range);
  `stats.yaml.iqr_arcsinh` gives you the per-channel scale. Apply `x' = arcsinh(x / s_c)` per channel c, then a final
  affine to ~unit variance. Validate the exact transform against stats.yaml before training; document your choice.

## 3. Target VAE interface (must hold so Stage 3 can consume it later)

- Output latent: **16 channels**, **16× spatial** (H/16, W/16), **4× temporal** (T/4). This matches
  `latent_channel_size=16`, `latent_downsample_factor=16`, `temporal_compression_factor=4`, `patch_spatial=2`
  from `cosmos_framework/configs/base/experiment/sft/models/nano_model_config.py`.
- For 88×256×2400 input → latent ≈ `[16, 22, 16, 150]` per trajectory. (Whole-dataset latent cache is tiny, ~3 GB.)
- **Pilot caveat:** physics fidelity DOMINATES this interface. If 16× spatial smears the shock front at small-r,
  it is acceptable to report that and recommend **8× spatial** for the real run (costs Stage 3 ~2× more tokens).
  The pilot's JOB is to measure this, not to force 16×.

## 4. Architecture to adapt (don't write a VAE from blank)

Adapt `cosmos_framework/model/vfm/tokenizers/wan2pt2_vae_4x16x16.py` (Wan2.2 VAE, ~100–150M params, dim=128,
dim_mult=[1,2,4,4], num_res_blocks=2, CausalConv3d-based 3D autoencoder).

KEY DETAIL — the Wan VAE patchifies 2× BEFORE the encoder:
- `Encoder3d.conv1 = CausalConv3d(12, dims[0], ...)`  → 12 = **3 RGB × 4** (2×2 space-to-depth). Encoder then does
  8× spatial / 4× temporal on the pre-reduced input → net 16× spatial / 4× temporal.
- `Decoder3d` final `CausalConv3d(out_dim, 12, ...)` → unpatchify back to 3.
- **Adaptation for 7 fields:** input patchify 7×4 = **28** channels → `conv1 = CausalConv3d(28, dims[0], ...)`;
  decoder out → 28 → unpatchify to 7. Set **`z_dim=16`**. Keep the rest of the topology.
- It's a plain autoencoder/VAE; for a tokenizer a small KL or just an AE + GAN-free L1/L2 recon is fine for the pilot.
  Recommended pilot loss: per-channel weighted L1 + a gradient/edge term (to protect shock fronts) + small KL.
  Do NOT add an adversarial/LPIPS term in the pilot (LPIPS is RGB-natural-image-trained; meaningless for fields).

## 5. Physics-fidelity priorities (USER'S EXPLICIT EMPHASIS — this is the point of the pilot)

The user stressed: *"the important thing is capturing all the physics... the laser heating is appropriate etc."*
So the eval is physics-first, not pixel-PSNR-first:

1. **All 7 channels must reconstruct** — report per-channel error separately (do NOT average into one number that
   hides a dead channel). Weight the loss per channel so low-magnitude-but-important fields (e.g. Tr) aren't ignored.
2. **Shock / ablation fronts** — the steep radial gradients (small-r implosion core, moving fronts) are the hard part
   and the whole reason 16× compression is risky. Eval MUST include a gradient-domain / front-location metric
   (e.g. error in dTe/dr peak location, or error on high-|∇| pixels specifically), not just bulk MSE.
3. **Laser heating** — use `laser_power[t]` to find timesteps where the drive is on / peaking, and verify the
   temperature fields (Te,Ti,Tr) reconstruct the heating front at those times. A VAE that's great in quiescent
   frames but smears the laser-driven front is a FAIL.
4. **Conservation sanity** — check that integrated mass (∫rho dV) and energy proxies are approximately preserved
   pre/post reconstruction (polar volume element ∝ r dr dθ — use coords_r/coords_theta). Large drift = fail.

### Cropping / temporal-window decision — DELEGATED TO YOU. Guidance & default:
- **Do NOT randomly crop in r naively** — the physics (fronts, core) lives at specific moving radii; a blind crop can
  cut the action. Prefer either (a) **full-r training** (W=2400 kept whole; tile/crop only in θ if memory forces it),
  or (b) **overlapping radial windows** that always include the high-gradient region, with overlap ≥ kernel receptive
  field to avoid seam artifacts. The polar geometry means r=0 (core) is special — keep it.
- **Temporal:** 88 frames, 4× temporal compression → use windows that are multiples giving clean latent frames
  (e.g. 16 or 24 frames → 4 or 6 latent frames). Causal conv VAE handles arbitrary T; pick a window long enough to
  span a front-crossing event but short enough to fit memory. Suggest 16-frame windows for the pilot.
- **Default starting point (change if your memory/quality analysis says so):** full θ (256), full r (2400),
  16-frame temporal windows, batch size tuned to fit 80GB with gradient checkpointing. If full-r×full-θ×16T doesn't
  fit, drop temporal window before you start cutting r. Document whatever you choose and WHY in the results.

## 6. Pilot plan (what to actually build & run this session)

Subset: take **~50 train trajectories** (and ~8 valid for eval). Pick deterministically (e.g. first 50 by id).

Steps:
1. **Data pipeline** (`fusion_posttrain/data/`): zarr→torch Iterable/Dataset reading from GCS (`gcsfs`+`zarr` v3) or
   pre-download the 50 stores locally (≈49 GB — fine). Apply per-channel arcsinh normalization from stats.yaml.
   Emit clips per §5 windowing. Verify a sample batch shape/stats BEFORE training (print min/max/mean per channel).
2. **Model** (`fusion_posttrain/model/field_vae.py`): adapt wan2pt2 VAE → 28-in/28-out, z_dim=16, 16×/4×.
3. **Train loop** (`fusion_posttrain/train_field_vae.py`): DDP across 4 GPUs, bf16, grad checkpointing,
   per-channel weighted L1 + gradient/edge term + small KL. Log per-channel recon error each eval.
4. **Recon eval** (`fusion_posttrain/eval_recon.py`): on held-out valid trajectories, produce the §5 metrics +
   side-by-side field heatmaps (input vs recon vs error) at laser-on timesteps and at front-crossing frames.
   Save PNGs to `fusion_posttrain/outputs/recon/`. These plots ARE the go/no-go evidence.
5. **Write `fusion_posttrain/RESULTS_01.md`**: per-channel numbers, the front/laser/conservation verdicts, the
   chosen windowing + why, and a clear **GO (16× holds) / GO-AT-8× / NO-GO** recommendation for Stage 3.

## 7. Infra — 4×A100-80GB spot via SkyPilot (cost-effective; do NOT use H100 for the VAE)

- The VAE is ~100–150M params → fits ONE 80GB A100 easily with tiling; 4×A100 is for wall-clock (DDP, ~linear).
- Use **SkyPilot managed/spot jobs** (`sky jobs launch`) for auto-recovery on preemption.
- Write `fusion_posttrain/sky_field_vae.yaml`: `resources: {accelerators: A100-80GB:4, use_spot: true}`,
  a `setup:` that `uv sync` (or pip installs torch+zarr+gcsfs) the training extras, and a `run:` that pulls the
  50-traj subset and launches `torchrun --nproc_per_node=4 fusion_posttrain/train_field_vae.py ...`.
- Pilot budget: ~1 GPU-day of work → ~6h wall on 4×A100 → est. **$30–60 spot**. If it's clearly passing/failing
  early, stop early.

## 8. UNVERIFIED — check these FIRST before building (env was never confirmed; the check was interrupted)

Run these and adapt:
- `sky --version` / `sky check` — is SkyPilot installed & is a cloud (GCP?) enabled with quota for A100-80GB:4 spot?
  If not installed: `uv pip install "skypilot[gcp]"` (or the relevant cloud) and `sky check`.
- `python -c "import torch,zarr,gcsfs"` — confirm deps locally (for building/debugging the pipeline before cloud).
- `gcloud auth list` / `gcloud config get-value project` — confirm creds can read `gs://zen_ml_data/...`
  (we already listed it successfully with `gsutil`, so read access exists from this box; confirm the SAME identity
  is available on the SkyPilot worker, or pass a service-account key).
- Confirm the exact key set in `stats.yaml` (we saw iqr/iqr_arcsinh/iqr_log10/iqr_log1p/max/min; check for mean/std).
- Decide local-dev vs cloud-only: you can build + smoke-test the pipeline/model on CPU or a tiny tensor locally,
  then launch the real run on SkyPilot. Do a 1-step dry run before the full job.

## 9. Definition of done for THIS session

- `fusion_posttrain/` contains: data pipeline, field_vae model, train script, eval script, SkyPilot YAML.
- A field-VAE pilot has been TRAINED on ~50 trajectories on 4×A100 spot (or a clearly-explained smaller fallback).
- `fusion_posttrain/RESULTS_01.md` exists with per-channel recon error, front/laser/conservation verdicts, the
  windowing decision, recon PNGs referenced, and a GO / GO-AT-8× / NO-GO call on 16× compression for Stage 3.
- DO NOT start the DiT SFT. That is Stage 3, a future session, gated on this result.

## 10. Pointers (verified file locations)

- Cosmos VAE to adapt: `cosmos_framework/model/vfm/tokenizers/wan2pt2_vae_4x16x16.py`
- DiT latent interface (for Stage-3 compatibility): `cosmos_framework/model/vfm/mot/cosmos3_vfm_network.py`
  (`patchify_and_pack_latents`, `vae2llm`/`llm2vae`, `latent_channel_size=16`)
- Nano model config: `cosmos_framework/configs/base/experiment/sft/models/nano_model_config.py`
- Stock Nano SFT recipe (the Stage-3 target, for reference only): `examples/toml/sft_config/vision_sft_nano.toml`,
  `examples/launch_sft_vision_nano.sh`
- Post-training skill (Stage-3 mechanics): `.claude/skills/cosmos3-post-training/SKILL.md`
- Custom-dataset / DataPacker docs: `docs/custom_dataset.md`, `docs/dataset_jsonl.md`
