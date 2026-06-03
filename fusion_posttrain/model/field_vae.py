# SPDX-License-Identifier: OpenMDW-1.1
"""Field VAE for GORGON fusion fields (Option C, Stage-1 pilot).

A clean, self-contained Wan2.2-*style* 3D ResBlock VAE. We do NOT vendor the
streaming-cache Wan tokenizer class (cosmos_framework.../wan2pt2_vae_4x16x16.py):
its temporal up/down convs only run on the feat_cache (inference) path, so the
plain full-forward path used for from-scratch training has an ambiguous temporal
geometry. The ONLY thing Stage-3 requires from the tokenizer is the latent
*interface* -- 16 channels, 16x spatial, 4x temporal, patch_spatial=2 compatible
-- not Wan's internal block identity. So we reproduce the exact compression
geometry with standard, shape-verified blocks.

Geometry (matches nano_model_config: latent_channel_size=16,
latent_downsample_factor=16, temporal_compression_factor=4):
  input  pixels [B, 7, T,  H,   W]   (7 physical fields)
  patch2 -> space-to-depth 2x on (H,W): [B, 28, T, H/2, W/2]
  encoder 8x spatial / 4x temporal:    [B, 16, T/4, H/16, W/16]   (z_dim=16)
  decoder mirror -> [B, 28, T, H/2, W/2] -> depth-to-space -> [B, 7, T, H, W]

For 16-frame, full-res GORGON clips: [B,7,16,256,2400] -> z [B,16,4,16,150].
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ---------------------------------------------------------------------------
# space-to-depth / depth-to-space on the spatial dims of a 5D [B,C,T,H,W] tensor
# ---------------------------------------------------------------------------
def space_to_depth(x: torch.Tensor, p: int = 2) -> torch.Tensor:
    b, c, t, h, w = x.shape
    x = x.view(b, c, t, h // p, p, w // p, p)
    x = x.permute(0, 1, 4, 6, 2, 3, 5).contiguous()  # [B,C,p,p,T,H/p,W/p]
    return x.view(b, c * p * p, t, h // p, w // p)


def depth_to_space(x: torch.Tensor, p: int = 2) -> torch.Tensor:
    b, cp, t, h, w = x.shape
    c = cp // (p * p)
    x = x.view(b, c, p, p, t, h, w)
    x = x.permute(0, 1, 4, 5, 2, 6, 3).contiguous()  # [B,C,T,H,p,W,p]
    return x.view(b, c, t, h * p, w * p)


def _gn(ch: int) -> nn.GroupNorm:
    g = 32
    while ch % g != 0:
        g //= 2
    return nn.GroupNorm(g, ch, eps=1e-6)


class ResBlock3d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.norm1 = _gn(in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.norm2 = _gn(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Down(nn.Module):
    """Strided downsample. stride=(st,2,2): st=2 also halves time."""

    def __init__(self, in_ch: int, out_ch: int, temporal: bool):
        super().__init__()
        st = 2 if temporal else 1
        self.conv = nn.Conv3d(in_ch, out_ch, 3, stride=(st, 2, 2), padding=1)

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    """Nearest-neighbour upsample + conv. temporal=True also doubles time."""

    def __init__(self, in_ch: int, out_ch: int, temporal: bool):
        super().__init__()
        self.scale = (2 if temporal else 1, 2, 2)
        self.conv = nn.Conv3d(in_ch, out_ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x.float(), scale_factor=self.scale, mode="nearest").type_as(x)
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, in_ch, base, ch_mult, z_dim, nblocks, use_checkpoint):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        dims = [base * m for m in ch_mult]            # e.g. [128,256,512,512]
        # transitions between the 4 levels: spatial always, temporal on last two
        temporal_flags = [False, True, True]
        self.conv_in = nn.Conv3d(in_ch, dims[0], 3, padding=1)
        self.blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i in range(len(dims)):
            level = nn.ModuleList(
                ResBlock3d(dims[i], dims[i]) for _ in range(nblocks)
            )
            self.blocks.append(level)
            if i < len(dims) - 1:
                self.downs.append(Down(dims[i], dims[i + 1], temporal_flags[i]))
        self.mid = nn.ModuleList([ResBlock3d(dims[-1], dims[-1]), ResBlock3d(dims[-1], dims[-1])])
        self.norm_out = _gn(dims[-1])
        self.conv_out = nn.Conv3d(dims[-1], 2 * z_dim, 3, padding=1)

    def _run(self, m, x):
        if self.use_checkpoint and self.training:
            return checkpoint(m, x, use_reentrant=False)
        return m(x)

    def forward(self, x):
        x = self.conv_in(x)
        for i, level in enumerate(self.blocks):
            for blk in level:
                x = self._run(blk, x)
            if i < len(self.downs):
                x = self.downs[i](x)
        for blk in self.mid:
            x = self._run(blk, x)
        x = self.conv_out(F.silu(self.norm_out(x)))
        return x  # [B, 2*z, T/4, H/16, W/16]


class Decoder(nn.Module):
    def __init__(self, out_ch, base, ch_mult, z_dim, nblocks, use_checkpoint):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        dims = [base * m for m in ch_mult]            # [128,256,512,512]
        rev = dims[::-1]                              # [512,512,256,128]
        temporal_flags = [True, True, False]         # mirror of encoder (reversed)
        self.conv_in = nn.Conv3d(z_dim, rev[0], 3, padding=1)
        self.mid = nn.ModuleList([ResBlock3d(rev[0], rev[0]), ResBlock3d(rev[0], rev[0])])
        self.ups = nn.ModuleList()
        self.blocks = nn.ModuleList()
        for i in range(len(rev) - 1):
            self.ups.append(Up(rev[i], rev[i + 1], temporal_flags[i]))
            level = nn.ModuleList(
                ResBlock3d(rev[i + 1], rev[i + 1]) for _ in range(nblocks)
            )
            self.blocks.append(level)
        self.norm_out = _gn(rev[-1])
        self.conv_out = nn.Conv3d(rev[-1], out_ch, 3, padding=1)

    def _run(self, m, x):
        if self.use_checkpoint and self.training:
            return checkpoint(m, x, use_reentrant=False)
        return m(x)

    def forward(self, x):
        x = self.conv_in(x)
        for blk in self.mid:
            x = self._run(blk, x)
        for i, up in enumerate(self.ups):
            x = up(x)
            for blk in self.blocks[i]:
                x = self._run(blk, x)
        x = self.conv_out(F.silu(self.norm_out(x)))
        return x


class FieldVAE(nn.Module):
    def __init__(
        self,
        in_channels: int = 7,
        z_dim: int = 16,
        base: int = 128,
        ch_mult=(1, 2, 4, 4),
        nblocks: int = 2,
        patch: int = 2,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.patch = patch
        self.z_dim = z_dim
        self.in_channels = in_channels
        pc = in_channels * patch * patch
        self.encoder = Encoder(pc, base, ch_mult, z_dim, nblocks, use_checkpoint)
        self.decoder = Decoder(pc, base, ch_mult, z_dim, nblocks, use_checkpoint)

    def encode(self, x):
        x = space_to_depth(x, self.patch)
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=1)
        return mu, logvar

    def decode(self, z):
        x = self.decoder(z)
        return depth_to_space(x, self.patch)

    def reparam(self, mu, logvar):
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar.clamp(-30, 20))
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        recon = self.decode(z)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon, mu, logvar, kl


if __name__ == "__main__":
    # quick self-test of shapes / param count (CPU, tiny)
    m = FieldVAE(use_checkpoint=False)
    n = sum(p.numel() for p in m.parameters())
    print(f"params: {n/1e6:.1f}M")
    x = torch.randn(1, 7, 16, 64, 96)
    mu, lv = m.encode(x)
    print("latent:", tuple(mu.shape), "expect (1,16,4,4,6)")
    r = m.decode(mu)
    print("recon:", tuple(r.shape), "expect", tuple(x.shape))
    assert r.shape == x.shape, (r.shape, x.shape)
    print("OK")
