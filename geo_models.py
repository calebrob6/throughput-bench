"""Geospatial foundation model wrappers for throughput benchmarking.

Provides a uniform nn.Module interface (forward(x) -> tensor) for encoders from:
- geobreeze (DOFA, CROMA, SenPaMAE, Galileo)
- olmoearth (OlmoEarth v1 Nano/Tiny/Base/Large)

Each wrapper creates the model architecture with random weights (no pretrained
checkpoint download) and normalizes the forward pass to accept a standard
(B, C, H, W) tensor. Per-model precision support is declared in
``GEO_MODEL_REGISTRY[name]['supported_precisions']``.
"""

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# geobreeze wrappers
# ---------------------------------------------------------------------------


class DOFAWrapper(nn.Module):
    """DOFA (Domain-adaptive Foundation model for geospatial Analysis).

    Uses wavelength-aware dynamic convolutions. Architecture from geobreeze.

    Precision: native ``fp16`` / ``bf16`` are not supported because DOFA's
    ``wave_dynamic_layer`` hardcodes ``torch.float32`` for the dynamic conv
    weights. ``amp`` works (weights stay fp32 under autocast) and is the
    recommended way to get half-precision throughput for this model.
    """

    def __init__(self, size: str = "base", num_channels: int = 3):
        super().__init__()
        from geobreeze.models.DOFA.models_dwv import (
            vit_base_patch16,
            vit_large_patch16,
        )

        if size == "base":
            self.encoder = vit_base_patch16()
        elif size == "large":
            self.encoder = vit_large_patch16()
        else:
            raise ValueError(f"Unknown DOFA size: {size}")

        default_waves = [
            0.443,
            0.490,
            0.560,
            0.665,
            0.705,
            0.740,
            0.783,
            0.842,
            0.865,
            0.945,
            1.375,
            1.610,
            2.190,
        ]
        self._wave_values = default_waves[:num_channels]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        waves = torch.tensor(self._wave_values, device=x.device, dtype=torch.float32)
        return self.encoder.forward_features(x, waves)


class CROMAWrapper(nn.Module):
    """CROMA (Cross-Modality Remote sensing representation learning).

    Custom ViT encoder for optical (12ch) or SAR (2ch) data.
    """

    def __init__(self, modality: str = "optical", image_resolution: int = 120):
        super().__init__()
        from geobreeze.models.croma import PretrainedCROMA

        self.encoder = PretrainedCROMA(
            pretrained_path=None,
            size="base",
            modality=modality,
            image_resolution=image_resolution,
        )
        self.modality = modality

        # Fix missing out_indices attribute in BaseTransformer
        for enc_attr in ("s2_encoder", "s1_encoder"):
            enc = getattr(self.encoder, enc_attr, None)
            if enc is not None and hasattr(enc, "transformer"):
                xf = enc.transformer
                if not hasattr(xf, "out_indices"):
                    xf.out_indices = [len(xf.layers) - 1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.modality == "optical":
            out = self.encoder(optical_images=x)
            return out["optical_GAP"]
        else:
            out = self.encoder(SAR_images=x)
            return out["SAR_GAP"]


class GalileoWrapper(nn.Module):
    """Galileo — time-aware ViT for Sentinel data.

    Uses geobreeze's Galileo encoder with official model configs from
    nasaharvest/galileo (Nano/Base/Large).
    """

    CONFIGS = {
        "nano": dict(embedding_size=128, depth=4, num_heads=8, mlp_ratio=4),
        "base": dict(embedding_size=768, depth=12, num_heads=12, mlp_ratio=4),
        "large": dict(embedding_size=1280, depth=24, num_heads=16, mlp_ratio=4),
    }

    def __init__(self, size: str = "base", input_key: str = "s2"):
        super().__init__()
        from geobreeze.models.galileo_src.model import Encoder

        if size not in self.CONFIGS:
            raise ValueError(f"Unknown Galileo size: {size}")

        self.encoder = Encoder(max_patch_size=8, max_sequence_length=1, **self.CONFIGS[size])
        self.input_key = input_key

        from geobreeze.models.galileo import Galileo as GalileoModel

        self._format_input = GalileoModel.format_input

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        formatted = self._format_input(self, x, self.input_key)
        out = self.encoder(patch_size=8, **formatted)
        s_t_tokens = out[0]  # (B, pH, pW, T, bands, embed)
        return s_t_tokens.mean(dim=(1, 2, 3, 4))


class SenPaMAEWrapper(nn.Module):
    """SenPaMAE — ViT-B/16 with spectral response function awareness.

    For benchmarking, provides dummy SRF (2301-dim) and GSD data.
    """

    def __init__(self, image_size: int = 144, num_channels: int = 3):
        super().__init__()
        from geobreeze.models.senpamae import vit_base_patch16

        self.encoder = vit_base_patch16(image_size=image_size, num_channels=num_channels)
        self.num_channels = num_channels
        # SRF encoding expects (B, C, 2301) — spectral response function samples
        self.register_buffer(
            "dummy_rf",
            torch.randn(1, num_channels, 2301),
        )
        # GSD encoding expects (B, C)
        self.register_buffer(
            "dummy_gsd",
            torch.ones(1, num_channels) * 10.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        rf = self.dummy_rf.expand(B, -1, -1)
        gsd = self.dummy_gsd.expand(B, -1)
        out = self.encoder(x, rf, gsd)
        # Returns (num_patches, B, embed_dim), mask_info — take CLS token
        features = out[0]  # (num_patches, B, embed_dim)
        return features[0]  # CLS token: (B, embed_dim)


# ---------------------------------------------------------------------------
# olmoearth wrappers
# ---------------------------------------------------------------------------


class OlmoEarthWrapper(nn.Module):
    """OlmoEarth v1 encoder wrapper for Sentinel-2 L2A (12 bands).

    Only the encoder is registered as a submodule, so `parameters()` reports
    encoder-only counts; the full `OlmoEarthPretrain_v1` also wraps an
    MAE-style decoder and a momentum target_encoder that aren't used for
    inference.

    The encoder is initialized with the default `supported_modality_names`
    (all 9 modalities the model was trained on) and `max_sequence_length=12`,
    matching paper Table 1 sizes (Nano 1.4M / Tiny 6.2M / Base 90M /
    Large 300M). Forward only feeds S2 + one timestep, so MACs reflect the
    actual single-modality inference path — patch embeddings for unused
    modalities sit idle and contribute zero FLOPs.
    """

    def __init__(self, model_size: str = "base"):
        super().__init__()
        from olmoearth_pretrain_minimal import OlmoEarthPretrain_v1

        full = OlmoEarthPretrain_v1(
            model_size=model_size,
            max_patch_size=8,
        )
        # Drop the pretraining decoder + target_encoder; we only want the encoder.
        self.encoder = full.model.encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes import (
            MaskedOlmoEarthSample,
        )

        B, C, H, W = x.shape
        # Convert (B, C, H, W) → (B, H, W, T=1, C=12)
        x_bhwtc = x.permute(0, 2, 3, 1).unsqueeze(3)

        timestamps = torch.zeros(B, 1, 3, dtype=torch.long, device=x.device)
        # Mask: (B, H, W, T, num_bandsets=3) — all visible
        mask = torch.zeros(B, H, W, 1, 3, dtype=torch.bool, device=x.device)

        sample = MaskedOlmoEarthSample(
            timestamps=timestamps,
            sentinel2_l2a=x_bhwtc,
            sentinel2_l2a_mask=mask,
        )

        out = self.encoder(sample, patch_size=8, input_res=10, fast_pass=True)
        tam = out["tokens_and_masks"]
        # Pool spatially: average the S2 token embeddings
        s2_tokens = tam.sentinel2_l2a  # (B, pH, pW, T, bandsets, embed)
        return s2_tokens.mean(dim=(1, 2, 3, 4))  # (B, embed_dim)


class OlmoEarthV1_1Wrapper(nn.Module):
    """OlmoEarth v1.1 encoder wrapper for Sentinel-2 L2A (12 bands).

    Same input/output contract as ``OlmoEarthWrapper`` — a ``(B, 12,
    128, 128)`` tensor in, a ``(B, embed_dim)`` pooled S2 embedding out.
    Built by going through ``load_model_from_id`` so the architecture
    exactly matches the released checkpoint configs on Hugging Face
    (``allenai/OlmoEarth-v1_1-{Nano,Tiny,Base}``). The raw
    ``OlmoEarthPretrain_v1(model_version="v1.1")`` constructor uses
    default-ish hyperparameters that don't quite match the released
    artifacts; going through HF guarantees matching encoder param
    counts (Nano 1.7M / Tiny 12.5M / Base 114M) whether or not weights
    are loaded.

    Compared to v1, v1.1 uses a single S2 bandset (12 bands in one
    group) instead of three, so the mask is ``(B, H, W, T, 1)`` and the
    forward output token tensor has ``bandsets=1`` in its fifth dim.
    There is no v1.1 ``Large`` checkpoint.

    Args:
        model_size: One of ``nano``, ``tiny``, ``base``.
        pretrained: When True (default), download the HF weights on
            first use and cache them under ``~/.cache/huggingface``.
            When False, build the same architecture from the HF config
            with random init — useful for throughput-only benchmarks
            that don't want a ~360 MB download for Base.
    """

    _MODEL_IDS = {
        "nano": "OLMOEARTH_V1_1_NANO",
        "tiny": "OLMOEARTH_V1_1_TINY",
        "base": "OLMOEARTH_V1_1_BASE",
    }

    def __init__(self, model_size: str = "base", pretrained: bool = True):
        super().__init__()
        from olmoearth_pretrain_minimal.model_loader import (
            ModelID,
            load_model_from_id,
        )

        if model_size not in self._MODEL_IDS:
            raise ValueError(
                f"Unknown OlmoEarth v1.1 size: {model_size!r}. "
                f"Must be one of {list(self._MODEL_IDS)}"
            )
        model_id = getattr(ModelID, self._MODEL_IDS[model_size])
        full = load_model_from_id(model_id, load_weights=pretrained)
        # Drop the pretraining decoder + target_encoder; we only want the encoder.
        self.encoder = full.encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes import (
            MaskedOlmoEarthSample,
        )

        B, C, H, W = x.shape
        # (B, C, H, W) → (B, H, W, T=1, C=12)
        x_bhwtc = x.permute(0, 2, 3, 1).unsqueeze(3)

        timestamps = torch.zeros(B, 1, 3, dtype=torch.long, device=x.device)
        # v1.1 collapses S2 into a single bandset, so num_bandsets=1.
        mask = torch.zeros(B, H, W, 1, 1, dtype=torch.bool, device=x.device)

        sample = MaskedOlmoEarthSample(
            timestamps=timestamps,
            sentinel2_l2a=x_bhwtc,
            sentinel2_l2a_mask=mask,
        )

        out = self.encoder(sample, patch_size=8, input_res=10, fast_pass=True)
        tam = out["tokens_and_masks"]
        s2_tokens = tam.sentinel2_l2a  # (B, pH, pW, T, bandsets=1, embed)
        return s2_tokens.mean(dim=(1, 2, 3, 4))  # (B, embed_dim)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

# Per-model native precision support. "fp32"/"fp16"/"bf16" cast the model with
# .float()/.half()/.bfloat16() and feed inputs of that dtype; "amp" leaves the
# model in fp32 and wraps forward in torch.autocast("cuda"). DOFA/CROMA/Galileo
# can't do native fp16/bf16 (upstream code mixes fp32 ops with half tensors)
# but they all work under AMP. OlmoEarth + SenPaMAE support all four.
ALL_PRECISIONS = frozenset({"fp32", "fp16", "bf16", "amp"})
FP32_AND_AMP = frozenset({"fp32", "amp"})

GEO_MODEL_REGISTRY: dict[str, dict] = {
    "dofa_base": {
        "cls": DOFAWrapper,
        "kwargs": {"size": "base", "num_channels": 3},
        "channels": 3,
        "size": 224,
        "family": "DOFA",
        "display": "DOFA-B/16",
        "params_approx": 111,
        "supported_precisions": FP32_AND_AMP,
    },
    "dofa_large": {
        "cls": DOFAWrapper,
        "kwargs": {"size": "large", "num_channels": 3},
        "channels": 3,
        "size": 224,
        "family": "DOFA",
        "display": "DOFA-L/16",
        "params_approx": 330,
        "supported_precisions": FP32_AND_AMP,
    },
    "croma_optical": {
        "cls": CROMAWrapper,
        "kwargs": {"modality": "optical", "image_resolution": 120},
        "channels": 12,
        "size": 120,
        "family": "CROMA",
        "display": "CROMA-Optical",
        "params_approx": 90,
        "supported_precisions": FP32_AND_AMP,
    },
    "croma_sar": {
        "cls": CROMAWrapper,
        "kwargs": {"modality": "SAR", "image_resolution": 120},
        "channels": 2,
        "size": 120,
        "family": "CROMA",
        "display": "CROMA-SAR",
        "params_approx": 90,
        "supported_precisions": FP32_AND_AMP,
    },
    "senpamae": {
        "cls": SenPaMAEWrapper,
        "kwargs": {"image_size": 144, "num_channels": 3},
        "channels": 3,
        "size": 144,
        "family": "SenPaMAE",
        "display": "SenPaMAE-B/16",
        "params_approx": 95,
        "supported_precisions": ALL_PRECISIONS,
    },
    "galileo_nano": {
        "cls": GalileoWrapper,
        "kwargs": {"size": "nano", "input_key": "s2"},
        "channels": 10,
        "size": 64,
        "family": "Galileo",
        "display": "Galileo-Nano/8",
        "params_approx": 1,
        "supported_precisions": FP32_AND_AMP,
    },
    "galileo_base": {
        "cls": GalileoWrapper,
        "kwargs": {"size": "base", "input_key": "s2"},
        "channels": 10,
        "size": 64,
        "family": "Galileo",
        "display": "Galileo-Base/8",
        "params_approx": 87,
        "supported_precisions": FP32_AND_AMP,
    },
    "galileo_large": {
        "cls": GalileoWrapper,
        "kwargs": {"size": "large", "input_key": "s2"},
        "channels": 10,
        "size": 64,
        "family": "Galileo",
        "display": "Galileo-Large/8",
        "params_approx": 380,
        "supported_precisions": FP32_AND_AMP,
    },
    # OlmoEarth fp16/bf16 require olmoearth_pretrain_minimal>=0.0.4 (PR #10).
    "olmoearth_nano": {
        "cls": OlmoEarthWrapper,
        "kwargs": {"model_size": "nano"},
        "channels": 12,
        "size": 128,
        "family": "OlmoEarth",
        "display": "OlmoEarth-Nano/8",
        "params_approx": 1.4,
        "supported_precisions": ALL_PRECISIONS,
    },
    "olmoearth_tiny": {
        "cls": OlmoEarthWrapper,
        "kwargs": {"model_size": "tiny"},
        "channels": 12,
        "size": 128,
        "family": "OlmoEarth",
        "display": "OlmoEarth-Tiny/8",
        "params_approx": 6.2,
        "supported_precisions": ALL_PRECISIONS,
    },
    "olmoearth_base": {
        "cls": OlmoEarthWrapper,
        "kwargs": {"model_size": "base"},
        "channels": 12,
        "size": 128,
        "family": "OlmoEarth",
        "display": "OlmoEarth-Base/8",
        "params_approx": 89,
        "supported_precisions": ALL_PRECISIONS,
    },
    "olmoearth_large": {
        "cls": OlmoEarthWrapper,
        "kwargs": {"model_size": "large"},
        "channels": 12,
        "size": 128,
        "family": "OlmoEarth",
        "display": "OlmoEarth-Large/8",
        "params_approx": 308,
        "supported_precisions": ALL_PRECISIONS,
    },
    # OlmoEarth v1.1 — pretrained weights downloaded from HF on first use
    # (allenai/OlmoEarth-v1_1-{Nano,Tiny,Base}). Requires
    # olmoearth_pretrain_minimal>=0.0.5. No Large checkpoint released.
    "olmoearth_v1_1_nano": {
        "cls": OlmoEarthV1_1Wrapper,
        "kwargs": {"model_size": "nano"},
        "channels": 12,
        "size": 128,
        "family": "OlmoEarth",
        "display": "OlmoEarth-v1.1-Nano/8",
        "params_approx": 1.7,
        "supported_precisions": ALL_PRECISIONS,
    },
    "olmoearth_v1_1_tiny": {
        "cls": OlmoEarthV1_1Wrapper,
        "kwargs": {"model_size": "tiny"},
        "channels": 12,
        "size": 128,
        "family": "OlmoEarth",
        "display": "OlmoEarth-v1.1-Tiny/8",
        "params_approx": 12.5,
        "supported_precisions": ALL_PRECISIONS,
    },
    "olmoearth_v1_1_base": {
        "cls": OlmoEarthV1_1Wrapper,
        "kwargs": {"model_size": "base"},
        "channels": 12,
        "size": 128,
        "family": "OlmoEarth",
        "display": "OlmoEarth-v1.1-Base/8",
        "params_approx": 114,
        "supported_precisions": ALL_PRECISIONS,
    },
}


def create_geo_model(name: str, device: torch.device) -> nn.Module:
    """Create a geo foundation model by registry name."""
    if name not in GEO_MODEL_REGISTRY:
        raise ValueError(f"Unknown geo model: {name}. Available: {list(GEO_MODEL_REGISTRY.keys())}")
    entry = GEO_MODEL_REGISTRY[name]
    model = entry["cls"](**entry["kwargs"])
    model = model.to(device)
    model.eval()
    return model
