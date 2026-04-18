"""Geospatial foundation model wrappers for throughput benchmarking.

Provides a uniform nn.Module interface (forward(x) -> tensor) for models from:
- geobreeze (DinoV2, DOFA, CROMA, SoftCon, Panopticon, Galileo, SenPaMAE, AnySat)
- olmoearth (OlmoEarth v1 Nano/Tiny/Base/Large)

Each wrapper creates the model architecture with random weights (no pretrained
checkpoint download) and normalizes the forward pass to accept a standard
(B, C, H, W) tensor.
"""

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# geobreeze wrappers
# ---------------------------------------------------------------------------


class DOFAWrapper(nn.Module):
    """DOFA (Domain-adaptive Foundation model for geospatial Analysis).

    Uses wavelength-aware dynamic convolutions. Architecture from geobreeze.
    Note: DOFA's wave_dynamic_layer hardcodes fp32 internally,
    so this model only runs in fp32 (fp16/bf16 are not supported).
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

        default_waves = [0.443, 0.490, 0.560, 0.665, 0.705, 0.740,
                         0.783, 0.842, 0.865, 0.945, 1.375, 1.610, 2.190]
        self._wave_values = default_waves[:num_channels]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        waves = torch.tensor(
            self._wave_values, device=x.device, dtype=torch.float32
        )
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

    # Official Galileo model configs
    CONFIGS = {
        "nano": dict(embedding_size=128, depth=4, num_heads=8, mlp_ratio=4),
        "base": dict(embedding_size=768, depth=12, num_heads=12, mlp_ratio=4),
        "large": dict(embedding_size=1280, depth=24, num_heads=16, mlp_ratio=4),
    }

    def __init__(
        self, size: str = "base", input_key: str = "s2", image_resolution: int = 64
    ):
        super().__init__()
        from geobreeze.models.galileo_src.model import Encoder

        if size not in self.CONFIGS:
            raise ValueError(f"Unknown Galileo size: {size}")

        self.encoder = Encoder(
            max_patch_size=8, max_sequence_length=1, **self.CONFIGS[size]
        )
        self.input_key = input_key
        self.image_resolution = image_resolution

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

        self.encoder = vit_base_patch16(
            image_size=image_size, num_channels=num_channels
        )
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


class AnySatWrapper(nn.Module):
    """AnySat — multi-modal ViT with modality-specific projectors.

    For benchmarking, uses Sentinel-2 modality (10 channels, 64×64).
    Requires torch.hub download of architecture code.
    """

    def __init__(self, input_key: str = "s2", image_resolution: int = 64):
        super().__init__()
        self.input_key = input_key
        self.image_resolution = image_resolution
        # AnySat requires torch.hub which downloads model code
        # We'll mark this as unavailable if it can't be loaded
        try:
            self.encoder = torch.hub.load(
                "gastruc/anysat", "anysat", pretrained=False
            )
        except Exception:
            raise RuntimeError(
                "AnySat requires torch.hub access to gastruc/anysat"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        # AnySat expects a dict with modality data
        data = {self.input_key: x.unsqueeze(1)}  # add temporal dim
        dates = {f"{self.input_key}_dates": torch.zeros(B, 1, device=x.device)}
        return self.encoder({**data, **dates})


# ---------------------------------------------------------------------------
# olmoearth wrappers
# ---------------------------------------------------------------------------


class OlmoEarthWrapper(nn.Module):
    """OlmoEarth v1 encoder wrapper.

    Converts standard (B, C, H, W) input to MaskedOlmoEarthSample for the encoder.
    Uses sentinel2_l2a modality (12 bands).
    """

    def __init__(self, model_size: str = "base", spatial_size: int = 128):
        super().__init__()
        from olmoearth_pretrain_minimal import OlmoEarthPretrain_v1

        self.model = OlmoEarthPretrain_v1(
            model_size=model_size,
            supported_modality_names=["sentinel2_l2a"],
            max_patch_size=8,
            max_sequence_length=1,
        )
        self.encoder = self.model.encoder
        self.spatial_size = spatial_size

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


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

# Registry of geo model constructors: name -> (wrapper_class, kwargs, native_channels, native_size)
GEO_MODEL_REGISTRY: dict[str, dict] = {
    # geobreeze models (fp32_only — custom architectures not designed for fp16)
    "dofa_base": {
        "cls": DOFAWrapper,
        "kwargs": {"size": "base", "num_channels": 3},
        "channels": 3,
        "size": 224,
        "family": "DOFA",
        "display": "DOFA-B/16",
        "params_approx": 111,
        "fp32_only": True,
    },
    "dofa_large": {
        "cls": DOFAWrapper,
        "kwargs": {"size": "large", "num_channels": 3},
        "channels": 3,
        "size": 224,
        "family": "DOFA",
        "display": "DOFA-L/16",
        "params_approx": 330,
        "fp32_only": True,
    },
    "croma_optical": {
        "cls": CROMAWrapper,
        "kwargs": {"modality": "optical", "image_resolution": 120},
        "channels": 12,
        "size": 120,
        "family": "CROMA",
        "display": "CROMA-Optical",
        "params_approx": 90,
        "fp32_only": True,
    },
    "croma_sar": {
        "cls": CROMAWrapper,
        "kwargs": {"modality": "SAR", "image_resolution": 120},
        "channels": 2,
        "size": 120,
        "family": "CROMA",
        "display": "CROMA-SAR",
        "params_approx": 90,
        "fp32_only": True,
    },
    "senpamae": {
        "cls": SenPaMAEWrapper,
        "kwargs": {"image_size": 144, "num_channels": 3},
        "channels": 3,
        "size": 144,
        "family": "SenPaMAE",
        "display": "SenPaMAE-B/16",
        "params_approx": 95,
        "fp32_only": True,
    },
    "galileo_nano": {
        "cls": GalileoWrapper,
        "kwargs": {"size": "nano", "input_key": "s2", "image_resolution": 64},
        "channels": 10,
        "size": 64,
        "family": "Galileo",
        "display": "Galileo-Nano",
        "params_approx": 1,
        "fp32_only": True,
    },
    "galileo_base": {
        "cls": GalileoWrapper,
        "kwargs": {"size": "base", "input_key": "s2", "image_resolution": 64},
        "channels": 10,
        "size": 64,
        "family": "Galileo",
        "display": "Galileo-Base",
        "params_approx": 87,
        "fp32_only": True,
    },
    "galileo_large": {
        "cls": GalileoWrapper,
        "kwargs": {"size": "large", "input_key": "s2", "image_resolution": 64},
        "channels": 10,
        "size": 64,
        "family": "Galileo",
        "display": "Galileo-Large",
        "params_approx": 380,
        "fp32_only": True,
    },
    # olmoearth models (fp32_only — custom FlexiViT with dtype issues in LayerNorm)
    "olmoearth_nano": {
        "cls": OlmoEarthWrapper,
        "kwargs": {"model_size": "nano", "spatial_size": 128},
        "channels": 12,
        "size": 128,
        "family": "OlmoEarth",
        "display": "OlmoEarth-Nano",
        "params_approx": 1.4,
        "fp32_only": True,
    },
    "olmoearth_tiny": {
        "cls": OlmoEarthWrapper,
        "kwargs": {"model_size": "tiny", "spatial_size": 128},
        "channels": 12,
        "size": 128,
        "family": "OlmoEarth",
        "display": "OlmoEarth-Tiny",
        "params_approx": 6.2,
        "fp32_only": True,
    },
    "olmoearth_base": {
        "cls": OlmoEarthWrapper,
        "kwargs": {"model_size": "base", "spatial_size": 128},
        "channels": 12,
        "size": 128,
        "family": "OlmoEarth",
        "display": "OlmoEarth-Base",
        "params_approx": 89,
        "fp32_only": True,
    },
    "olmoearth_large": {
        "cls": OlmoEarthWrapper,
        "kwargs": {"model_size": "large", "spatial_size": 128},
        "channels": 12,
        "size": 128,
        "family": "OlmoEarth",
        "display": "OlmoEarth-Large",
        "params_approx": 308,
        "fp32_only": True,
    },
}


def create_geo_model(name: str, device: torch.device) -> nn.Module:
    """Create a geo foundation model by registry name."""
    if name not in GEO_MODEL_REGISTRY:
        raise ValueError(
            f"Unknown geo model: {name}. "
            f"Available: {list(GEO_MODEL_REGISTRY.keys())}"
        )
    entry = GEO_MODEL_REGISTRY[name]
    model = entry["cls"](**entry["kwargs"])
    model = model.to(device)
    model.eval()
    return model
