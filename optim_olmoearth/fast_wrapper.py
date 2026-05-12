"""Fast inference path for OlmoEarth-Base on Sentinel-2 L2A only.

This module reproduces the exact forward output of
``geo_models.OlmoEarthWrapper(model_size="base")`` for a
``(B, 12, 128, 128)`` input — but skips the per-call dataclass
construction, modality dispatch, mask sorting, and runtime composite
encoding computation that dominate Python overhead in the reference
implementation.

Why this is equivalent
----------------------
For the wrapper's input contract we have, by construction:
    * Only Sentinel-2 L2A is fed (T=1, mask = all visible).
    * The 9 supported modalities other than S2 receive no data, so
      their patch-embed / channel-embed / pos-embed contributions are
      identical to the reference (zero — they're never invoked).
    * ``fast_pass=True`` already turns off mask construction in the
      reference; our path just removes the dead code entirely.
    * All channel embeddings start as ``torch.zeros(...)`` and are
      never touched by ``_init_weights`` (which only initializes
      ``nn.Linear``), so for the random-weights inference benchmark
      they're exactly zero.
    * Time pos embed at T=1 picks index 0; month embed at month 0; both
      are constants — pre-baked into a single additive bias tensor.

What changes
------------
1. Three ``nn.Conv2d`` patch embeddings collapse into one
   ``nn.Linear(12*64, 3*768)`` — equivalent matrix because each conv
   only reads its own bandset's channel indices, which we encode as
   a sparse-then-densified weight matrix. Linear hits cuBLAS GEMM, the
   conv hit cuDNN paths that are slow on small in_chans.
2. Three separate ``nn.Linear(dim, dim)`` for q/k/v collapse into one
   fused ``nn.Linear(dim, 3*dim)`` per attention block.
3. Composite spatial+temporal+month+channel encoding is pre-computed
   once at module init and stored as a buffer.
4. The final ``s2_tokens.mean(dim=(1, 2, 3, 4))`` over (h, w, T,
   bandsets) is identical to ``tokens.mean(dim=1)`` on the flattened
   sequence — we don't need to round-trip through the per-modality
   dict.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Fused attention block
# ---------------------------------------------------------------------------


class FusedBlock(nn.Module):
    """ViT block with fused QKV and SDPA, matching the reference Block exactly.

    Equivalent to ``olmoearth_pretrain_v1.nn.attention.Block`` for the
    inference-only path with: ``qk_norm=False``, ``init_values=None``
    (so LayerScale is Identity), ``drop_path=0`` *in eval mode*,
    ``cross_attn=False``, ``use_flash_attn=False``, and ``attn_mask=None``.
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        h = self.norm1(x)
        qkv = self.qkv(h)
        B, N, _ = qkv.shape
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0)
        a = a.transpose(1, 2).reshape(B, N, self.dim)
        x = x + self.proj(a)

        # MLP
        h = self.norm2(x)
        h = self.fc1(h)
        h = F.gelu(h)
        h = self.fc2(h)
        x = x + h
        return x


# ---------------------------------------------------------------------------
# Patch embedding
# ---------------------------------------------------------------------------


class FusedPatchEmbed(nn.Module):
    """Single Linear that replicates the 3 per-bandset Conv2d patch embeds.

    For S2 L2A on a (B, 12, 128, 128) input, the reference does:
      * idx_select bandset i's channels → (B, c_i, 128, 128)
      * Conv2d(c_i → 768, kernel=8, stride=8) → (B, 768, 16, 16)
      * stack along bandset dim → (B, 16, 16, T=1, 3, 768)

    We replace this with one Linear(768=12*64 → 3*768=2304) operating
    on flattened 8×8×12 patches. The three Conv2d weight tensors of
    shape (768, c_i, 8, 8) are sparse-stitched into a single
    (2304, 12*64) matrix that is *exactly* equivalent: rows
    bandset_i*768 .. (bandset_i+1)*768 take only the input columns
    corresponding to bandset_i's channel indices.
    """

    BAND_INDICES: tuple[tuple[int, ...], ...] = (
        (0, 1, 2, 3),       # bandset 0: 10 m bands  (B02, B03, B04, B08)
        (4, 5, 6, 7, 8, 9), # bandset 1: 20 m bands  (B05, B06, B07, B8A, B11, B12)
        (10, 11),           # bandset 2: 60 m bands  (B01, B09)
    )

    def __init__(self, embed_dim: int = 768, patch: int = 8, in_channels: int = 12):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch = patch
        self.in_channels = in_channels
        n_bandsets = len(self.BAND_INDICES)
        # in_features = patch * patch * in_channels (rearranged from (c, p, p))
        in_features = patch * patch * in_channels
        out_features = n_bandsets * embed_dim
        self.proj = nn.Linear(in_features, out_features, bias=True)

    @torch.no_grad()
    def absorb_from_reference(
        self,
        ref_per_modality_module_dict: nn.ModuleDict,
    ) -> None:
        """Copy the 3 reference Conv2d weights/biases into our fused Linear.

        ``ref_per_modality_module_dict`` is expected to be the
        ``per_modality_embeddings['sentinel2_l2a']`` ModuleDict from the
        reference encoder, with keys ``sentinel2_l2a__{0,1,2}`` whose
        ``.proj`` is ``nn.Conv2d``.

        Conv2d weight ``(out, c, p, p)`` stores patches in
        (channel, p_h, p_w) memory order, but our Linear takes
        ``rearrange(x, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)")`` —
        i.e. (p_h, p_w, channel). We permute the conv weight to
        ``(out, p_h, p_w, c)`` then flatten before placing into the
        right column slice for this bandset.
        """
        new_w = torch.zeros(
            self.proj.out_features,
            self.proj.in_features,
            dtype=self.proj.weight.dtype,
            device=self.proj.weight.device,
        )
        new_b = torch.zeros(
            self.proj.out_features,
            dtype=self.proj.bias.dtype,
            device=self.proj.bias.device,
        )

        embed = self.embed_dim
        p = self.patch
        ch = self.in_channels
        for bs_idx, channel_idxs in enumerate(self.BAND_INDICES):
            ref_mod = ref_per_modality_module_dict[f"sentinel2_l2a__{bs_idx}"]
            conv_w = ref_mod.proj.weight  # (embed, c_bs, p, p)
            conv_b = ref_mod.proj.bias    # (embed,)

            # (embed, c_bs, p, p) -> (embed, p, p, c_bs) -> (embed, p*p*c_bs)
            w_perm = conv_w.permute(0, 2, 3, 1).contiguous()

            # Place each output channel of conv into the linear weight matrix.
            # Linear input is laid out as (p_h, p_w, channel), where channel
            # ranges over ALL 12 input channels. We only fill the columns
            # corresponding to this bandset's channel indices.
            row_off = bs_idx * embed
            for out_i in range(embed):
                # w_perm[out_i] has shape (p, p, c_bs); place into a
                # (p, p, ch) zeroed tensor at the channel indices.
                slice_in = w_perm[out_i]  # (p, p, c_bs)
                full = torch.zeros(p, p, ch, dtype=conv_w.dtype, device=conv_w.device)
                for ci_local, ci_global in enumerate(channel_idxs):
                    full[:, :, ci_global] = slice_in[:, :, ci_local]
                new_w[row_off + out_i] = full.reshape(-1)
            new_b[row_off : row_off + embed] = conv_b

        self.proj.weight.copy_(new_w)
        self.proj.bias.copy_(new_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(B, 12, 128, 128)`` → ``(B, 3*256, 768)`` flattened tokens."""
        B, C, H, W = x.shape
        p = self.patch
        # (B, C, H, W) -> (B, num_h, num_w, p_h, p_w, C) -> (B, num_h*num_w, p_h*p_w*C)
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()  # (B, num_h, num_w, p, p, C)
        n_patches = (H // p) * (W // p)
        x = x.reshape(B, n_patches, p * p * C)
        x = self.proj(x)  # (B, n_patches, n_bandsets * embed)
        # Reshape to (B, n_patches, n_bandsets, embed) then flatten bandsets in
        # the same order as the reference's collapse (modality -> spatial ->
        # bandset). The reference does
        #   stack(tokens_per_bandset, dim=-2) -> (B, ph, pw, T=1, 3, D)
        # then collapse_and_combine_hwtc rearranges as
        #   "b ... d -> b (...) d"   i.e. (b, ph*pw*T*bs, d) with bandset as
        # the innermost dim. Mirroring that order:
        x = x.reshape(B, n_patches, len(self.BAND_INDICES), self.embed_dim)
        x = x.reshape(B, n_patches * len(self.BAND_INDICES), self.embed_dim)
        return x


# ---------------------------------------------------------------------------
# Pre-baked composite encoding bias
# ---------------------------------------------------------------------------


def _compute_constant_encoding_bias(ref_encoder, embed_dim: int = 768) -> torch.Tensor:
    """Compute the additive bias contributed by ``CompositeEncodings`` for
    our fixed (S2 L2A only, T=1, patch=8, input_res=10) input.

    The reference splits the embedding into 4 quarters of size
    ``embedding_dim_per_embedding_type = embed_dim // 4 = 192``:
      * [0   :  192] channel embedding (zeros at random init)
      * [192 :  384] time-position embedding (pos[0] at T=1)
      * [384 :  576] month embedding (month=0 at timestamps=zeros)
      * [576 :  768] 2D sin-cos spatial encoding at gsd_ratio=8

    All of these are constants per call; we precompute once and add.

    Returns a tensor of shape (1, 768=ph*pw*bandsets, embed_dim) with
    the bandset axis innermost — matching the FusedPatchEmbed token
    ordering.
    """
    n = embed_dim // 4
    device = next(ref_encoder.parameters()).device
    dtype = torch.float32

    composite = ref_encoder.composite_encodings

    # (1) channel embedding for sentinel2_l2a is (3, 192). Zeros at random
    # init, but copy whatever the reference holds (it may have been touched).
    channel = composite.per_modality_channel_embeddings["sentinel2_l2a"].to(
        device=device, dtype=dtype
    )  # (3, 192)

    # (2) time pos embed at index 0
    time_emb = composite.pos_embed[0].to(device=device, dtype=dtype)  # (192,)

    # (3) month embed at month=0
    month_emb = composite.month_embed.weight[0].to(device=device, dtype=dtype)  # (192,)

    # (4) spatial encoding at gsd_ratio = input_res * patch / BASE_GSD = 10*8/10 = 8
    from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.nn.encodings import (
        get_2d_sincos_pos_encoding_with_resolution,
    )
    h = w = 16
    spatial = get_2d_sincos_pos_encoding_with_resolution(
        grid_size=h,
        res=torch.ones(1, device=device) * 8.0,
        encoding_dim=n,
        device=device,
    )  # (1, h*w, 192)
    spatial = spatial.to(dtype)

    n_patches = h * w
    n_bs = channel.shape[0]
    bias = torch.zeros(n_patches, n_bs, embed_dim, device=device, dtype=dtype)
    bias[:, :, 0:n] = channel.unsqueeze(0)              # broadcast over patches
    bias[:, :, n : 2 * n] = time_emb                    # broadcast over (patches, bs)
    bias[:, :, 2 * n : 3 * n] = month_emb               # broadcast over (patches, bs)
    bias[:, :, 3 * n : 4 * n] = spatial[0].unsqueeze(1) # (n_patches, 1, n)

    # Flatten (patches, bandsets) in the same order as the FusedPatchEmbed
    # output: spatial is the outer dim, bandset is innermost.
    return bias.reshape(1, n_patches * n_bs, embed_dim)


# ---------------------------------------------------------------------------
# Top-level fast wrapper
# ---------------------------------------------------------------------------


class FastOlmoEarthBase(nn.Module):
    """Drop-in equivalent of ``geo_models.OlmoEarthWrapper("base")`` for
    ``(B, 12, 128, 128)`` Sentinel-2 L2A input.

    Built by calling ``FastOlmoEarthBase.from_reference(ref_wrapper)`` so
    weights are absorbed from a fully-initialized reference encoder.
    """

    EMBED_DIM = 768
    NUM_HEADS = 12
    DEPTH = 12
    MLP_RATIO = 4.0
    PATCH = 8
    IN_CHANNELS = 12

    def __init__(self):
        super().__init__()
        self.patch_embed = FusedPatchEmbed(
            embed_dim=self.EMBED_DIM, patch=self.PATCH, in_channels=self.IN_CHANNELS
        )
        self.blocks = nn.ModuleList(
            [
                FusedBlock(self.EMBED_DIM, self.NUM_HEADS, self.MLP_RATIO)
                for _ in range(self.DEPTH)
            ]
        )
        self.norm = nn.LayerNorm(self.EMBED_DIM)
        # Constant encoding bias — registered as a buffer so .half()/.bfloat16()
        # casts propagate to it.
        self.register_buffer(
            "encoding_bias",
            torch.zeros(1, 768, self.EMBED_DIM),
            persistent=False,
        )

    @classmethod
    @torch.no_grad()
    def from_reference(cls, ref_wrapper) -> "FastOlmoEarthBase":
        """Construct from a ``geo_models.OlmoEarthWrapper`` reference."""
        ref_encoder = ref_wrapper.encoder
        device = next(ref_encoder.parameters()).device
        dtype = next(ref_encoder.parameters()).dtype

        out = cls().to(device=device, dtype=dtype)

        # 1. Patch embed: absorb the 3 S2 conv weights into our fused linear.
        out.patch_embed.absorb_from_reference(
            ref_encoder.patch_embeddings.per_modality_embeddings["sentinel2_l2a"]
        )

        # 2. Per-block: copy LayerNorm, fuse q/k/v, copy proj/MLP.
        for fast_blk, ref_blk in zip(out.blocks, ref_encoder.blocks):
            fast_blk.norm1.weight.copy_(ref_blk.norm1.weight)
            fast_blk.norm1.bias.copy_(ref_blk.norm1.bias)

            qkv_w = torch.cat(
                [ref_blk.attn.q.weight, ref_blk.attn.k.weight, ref_blk.attn.v.weight],
                dim=0,
            )
            qkv_b = torch.cat(
                [ref_blk.attn.q.bias, ref_blk.attn.k.bias, ref_blk.attn.v.bias],
                dim=0,
            )
            fast_blk.qkv.weight.copy_(qkv_w)
            fast_blk.qkv.bias.copy_(qkv_b)

            fast_blk.proj.weight.copy_(ref_blk.attn.proj.weight)
            fast_blk.proj.bias.copy_(ref_blk.attn.proj.bias)

            fast_blk.norm2.weight.copy_(ref_blk.norm2.weight)
            fast_blk.norm2.bias.copy_(ref_blk.norm2.bias)

            fast_blk.fc1.weight.copy_(ref_blk.mlp.fc1.weight)
            fast_blk.fc1.bias.copy_(ref_blk.mlp.fc1.bias)
            fast_blk.fc2.weight.copy_(ref_blk.mlp.fc2.weight)
            fast_blk.fc2.bias.copy_(ref_blk.mlp.fc2.bias)

        # 3. Final norm
        out.norm.weight.copy_(ref_encoder.norm.weight)
        out.norm.bias.copy_(ref_encoder.norm.bias)

        # 4. Pre-bake the composite encoding bias.
        bias = _compute_constant_encoding_bias(ref_encoder, embed_dim=cls.EMBED_DIM)
        out.encoding_bias = bias.to(device=device, dtype=dtype)

        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(B, 12, 128, 128)`` → ``(B, 768)``."""
        tokens = self.patch_embed(x)        # (B, 768, 768)
        tokens = tokens + self.encoding_bias  # broadcast (1, 768, 768)
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        return tokens.mean(dim=1)
