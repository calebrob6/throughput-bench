"""Model registry for Throughput Bench benchmarks.

Each model entry defines the timm model name and display metadata. All
benchmarks are encoder-only classification — no decoders are attached.

`timm_name` doubles as the unique benchmark / CSV identifier. To benchmark
a single timm architecture under multiple configurations (e.g. the same
ViT-L body with different patch sizes), set `timm_factory_name` to the
real timm model name and use a synthetic `timm_name` as the benchmark ID.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    timm_name: str  # also used as the benchmark / CSV identifier
    display_name: str
    family: str
    arch_type: str  # "cnn", "vit", "hybrid"
    color: str
    source: str = "timm"  # "timm", "geo"
    native_channels: int = 3
    native_size: int = 224
    geo_model_key: str = ""  # key into GEO_MODEL_REGISTRY
    # Overrides for routing through timm.create_model — used when the
    # benchmark ID can't be a real timm model name (e.g. a patch-size
    # override that timm doesn't ship a pretrained checkpoint for).
    timm_factory_name: str = ""
    patch_size: int | None = None

    @property
    def timm_factory(self) -> str:
        """The actual timm model name to pass to `timm.create_model`."""
        return self.timm_factory_name or self.timm_name


# Family color palette
FAMILY_COLORS = {
    "ResNet": "#1f77b4",
    "EfficientNet": "#2ca02c",
    "ConvNeXt": "#9467bd",
    "MobileNet": "#8c564b",
    "RegNet": "#e377c2",
    "ViT": "#d62728",
    "DeiT": "#ff7f0e",
    "Swin": "#bcbd22",
    "BEiT": "#17becf",
    "CoAtNet": "#7f7f7f",
    "DinoV3": "#1B998B",  # dark teal — modern self-supervised ViTs
    # Geo foundation model families. Colors picked to be visually distinct
    # from the timm families above (no near-duplicate hues).
    "DOFA": "#e6550d",  # dark orange (distinct from DeiT's bright orange)
    "CROMA": "#117733",  # dark green (distinct from EfficientNet's medium green)
    "SenPaMAE": "#882E72",  # deep purple (distinct from ConvNeXt's lavender)
    "Galileo": "#C71585",  # magenta (distinct from ViT's red and RegNet's pink)
    "OlmoEarth": "#003F5C",  # navy (distinct from ResNet's steel blue)
}

MODEL_REGISTRY: list[ModelConfig] = [
    # ── CNNs ──────────────────────────────────────────────────────────────
    ModelConfig("resnet18", "ResNet-18", "ResNet", "cnn", FAMILY_COLORS["ResNet"]),
    ModelConfig("resnet50", "ResNet-50", "ResNet", "cnn", FAMILY_COLORS["ResNet"]),
    ModelConfig("resnet101", "ResNet-101", "ResNet", "cnn", FAMILY_COLORS["ResNet"]),
    ModelConfig("resnet152", "ResNet-152", "ResNet", "cnn", FAMILY_COLORS["ResNet"]),
    ModelConfig(
        "efficientnet_b0",
        "EfficientNet-B0",
        "EfficientNet",
        "cnn",
        FAMILY_COLORS["EfficientNet"],
    ),
    ModelConfig(
        "efficientnet_b4",
        "EfficientNet-B4",
        "EfficientNet",
        "cnn",
        FAMILY_COLORS["EfficientNet"],
    ),
    ModelConfig(
        "efficientnet_b7",
        "EfficientNet-B7",
        "EfficientNet",
        "cnn",
        FAMILY_COLORS["EfficientNet"],
    ),
    ModelConfig("convnext_tiny", "ConvNeXt-T", "ConvNeXt", "cnn", FAMILY_COLORS["ConvNeXt"]),
    ModelConfig("convnext_small", "ConvNeXt-S", "ConvNeXt", "cnn", FAMILY_COLORS["ConvNeXt"]),
    ModelConfig("convnext_base", "ConvNeXt-B", "ConvNeXt", "cnn", FAMILY_COLORS["ConvNeXt"]),
    ModelConfig("convnext_large", "ConvNeXt-L", "ConvNeXt", "cnn", FAMILY_COLORS["ConvNeXt"]),
    ModelConfig(
        "mobilenetv3_small_100",
        "MobileNetV3-S",
        "MobileNet",
        "cnn",
        FAMILY_COLORS["MobileNet"],
    ),
    ModelConfig(
        "mobilenetv3_large_100",
        "MobileNetV3-L",
        "MobileNet",
        "cnn",
        FAMILY_COLORS["MobileNet"],
    ),
    ModelConfig("regnety_004", "RegNetY-400MF", "RegNet", "cnn", FAMILY_COLORS["RegNet"]),
    ModelConfig("regnety_040", "RegNetY-4GF", "RegNet", "cnn", FAMILY_COLORS["RegNet"]),
    # ── Vision Transformers ───────────────────────────────────────────────
    ModelConfig(
        "vit_tiny_patch16_224",
        "ViT-Ti/16",
        "ViT",
        "vit",
        FAMILY_COLORS["ViT"],
    ),
    ModelConfig(
        "vit_small_patch16_224",
        "ViT-S/16",
        "ViT",
        "vit",
        FAMILY_COLORS["ViT"],
    ),
    ModelConfig(
        "vit_base_patch16_224",
        "ViT-B/16",
        "ViT",
        "vit",
        FAMILY_COLORS["ViT"],
    ),
    ModelConfig(
        "vit_large_patch16_224",
        "ViT-L/16",
        "ViT",
        "vit",
        FAMILY_COLORS["ViT"],
    ),
    ModelConfig(
        "vit_huge_patch14_224",
        "ViT-H/14",
        "ViT",
        "vit",
        FAMILY_COLORS["ViT"],
    ),
    ModelConfig(
        "vit_giant_patch14_224",
        "ViT-G/14",
        "ViT",
        "vit",
        FAMILY_COLORS["ViT"],
    ),
    # Synthetic ID — uses ViT-L/16's body but overrides patch_size to 8 for
    # an apples-to-apples compute comparison against OlmoEarth-Large/8 (same
    # depth × dim, same ~768 spatial-token count, same ~470 GMACs at 224²).
    ModelConfig(
        "vit_large_patch8_224",
        "ViT-L/8",
        "ViT",
        "vit",
        FAMILY_COLORS["ViT"],
        timm_factory_name="vit_large_patch16_224",
        patch_size=8,
    ),
    ModelConfig(
        "deit3_small_patch16_224",
        "DeiT3-S/16",
        "DeiT",
        "vit",
        FAMILY_COLORS["DeiT"],
    ),
    ModelConfig(
        "deit3_base_patch16_224",
        "DeiT3-B/16",
        "DeiT",
        "vit",
        FAMILY_COLORS["DeiT"],
    ),
    ModelConfig("swin_tiny_patch4_window7_224", "Swin-T", "Swin", "vit", FAMILY_COLORS["Swin"]),
    ModelConfig("swin_small_patch4_window7_224", "Swin-S", "Swin", "vit", FAMILY_COLORS["Swin"]),
    ModelConfig("swin_base_patch4_window7_224", "Swin-B", "Swin", "vit", FAMILY_COLORS["Swin"]),
    ModelConfig("swin_large_patch4_window7_224", "Swin-L", "Swin", "vit", FAMILY_COLORS["Swin"]),
    ModelConfig(
        "beit_base_patch16_224",
        "BEiT-B/16",
        "BEiT",
        "vit",
        FAMILY_COLORS["BEiT"],
    ),
    ModelConfig(
        "beit_large_patch16_224",
        "BEiT-L/16",
        "BEiT",
        "vit",
        FAMILY_COLORS["BEiT"],
    ),
    # ── DinoV3 (Meta self-supervised ViTs) ────────────────────────────────
    ModelConfig(
        "vit_huge_plus_patch16_dinov3",
        "DinoV3-H+/16",
        "DinoV3",
        "vit",
        FAMILY_COLORS["DinoV3"],
    ),
    # ── Hybrids ───────────────────────────────────────────────────────────
    ModelConfig("coatnet_0_224", "CoAtNet-0", "CoAtNet", "hybrid", FAMILY_COLORS["CoAtNet"]),
    ModelConfig("coatnet_2_224", "CoAtNet-2", "CoAtNet", "hybrid", FAMILY_COLORS["CoAtNet"]),
    # ── Geospatial Foundation Models (custom geobreeze / olmoearth wrappers) ─
    ModelConfig(
        "dofa_base",
        "DOFA-B/16",
        "DOFA",
        "vit",
        FAMILY_COLORS["DOFA"],
        source="geo",
        native_channels=3,
        native_size=224,
        geo_model_key="dofa_base",
    ),
    ModelConfig(
        "dofa_large",
        "DOFA-L/16",
        "DOFA",
        "vit",
        FAMILY_COLORS["DOFA"],
        source="geo",
        native_channels=3,
        native_size=224,
        geo_model_key="dofa_large",
    ),
    ModelConfig(
        "croma_optical",
        "CROMA-Optical",
        "CROMA",
        "vit",
        FAMILY_COLORS["CROMA"],
        source="geo",
        native_channels=12,
        native_size=120,
        geo_model_key="croma_optical",
    ),
    ModelConfig(
        "croma_sar",
        "CROMA-SAR",
        "CROMA",
        "vit",
        FAMILY_COLORS["CROMA"],
        source="geo",
        native_channels=2,
        native_size=120,
        geo_model_key="croma_sar",
    ),
    ModelConfig(
        "senpamae",
        "SenPaMAE-B/16",
        "SenPaMAE",
        "vit",
        FAMILY_COLORS["SenPaMAE"],
        source="geo",
        native_channels=3,
        native_size=144,
        geo_model_key="senpamae",
    ),
    ModelConfig(
        "galileo_nano",
        "Galileo-Nano/8",
        "Galileo",
        "vit",
        FAMILY_COLORS["Galileo"],
        source="geo",
        native_channels=10,
        native_size=64,
        geo_model_key="galileo_nano",
    ),
    ModelConfig(
        "galileo_base",
        "Galileo-Base/8",
        "Galileo",
        "vit",
        FAMILY_COLORS["Galileo"],
        source="geo",
        native_channels=10,
        native_size=64,
        geo_model_key="galileo_base",
    ),
    ModelConfig(
        "galileo_large",
        "Galileo-Large/8",
        "Galileo",
        "vit",
        FAMILY_COLORS["Galileo"],
        source="geo",
        native_channels=10,
        native_size=64,
        geo_model_key="galileo_large",
    ),
    ModelConfig(
        "olmoearth_nano",
        "OlmoEarth-Nano/8",
        "OlmoEarth",
        "vit",
        FAMILY_COLORS["OlmoEarth"],
        source="geo",
        native_channels=12,
        native_size=128,
        geo_model_key="olmoearth_nano",
    ),
    ModelConfig(
        "olmoearth_tiny",
        "OlmoEarth-Tiny/8",
        "OlmoEarth",
        "vit",
        FAMILY_COLORS["OlmoEarth"],
        source="geo",
        native_channels=12,
        native_size=128,
        geo_model_key="olmoearth_tiny",
    ),
    ModelConfig(
        "olmoearth_base",
        "OlmoEarth-Base/8",
        "OlmoEarth",
        "vit",
        FAMILY_COLORS["OlmoEarth"],
        source="geo",
        native_channels=12,
        native_size=128,
        geo_model_key="olmoearth_base",
    ),
    ModelConfig(
        "olmoearth_large",
        "OlmoEarth-Large/8",
        "OlmoEarth",
        "vit",
        FAMILY_COLORS["OlmoEarth"],
        source="geo",
        native_channels=12,
        native_size=128,
        geo_model_key="olmoearth_large",
    ),
    ModelConfig(
        "olmoearth_v1_1_nano",
        "OlmoEarth-v1.1-Nano/8",
        "OlmoEarth",
        "vit",
        FAMILY_COLORS["OlmoEarth"],
        source="geo",
        native_channels=12,
        native_size=128,
        geo_model_key="olmoearth_v1_1_nano",
    ),
    ModelConfig(
        "olmoearth_v1_1_tiny",
        "OlmoEarth-v1.1-Tiny/8",
        "OlmoEarth",
        "vit",
        FAMILY_COLORS["OlmoEarth"],
        source="geo",
        native_channels=12,
        native_size=128,
        geo_model_key="olmoearth_v1_1_tiny",
    ),
    ModelConfig(
        "olmoearth_v1_1_base",
        "OlmoEarth-v1.1-Base/8",
        "OlmoEarth",
        "vit",
        FAMILY_COLORS["OlmoEarth"],
        source="geo",
        native_channels=12,
        native_size=128,
        geo_model_key="olmoearth_v1_1_base",
    ),
]


def get_models(names: list[str] | None = None) -> list[ModelConfig]:
    """Return models filtered by timm name. If names is None, return all."""
    if names is None:
        return list(MODEL_REGISTRY)
    name_set = set(names)
    return [m for m in MODEL_REGISTRY if m.timm_name in name_set]
