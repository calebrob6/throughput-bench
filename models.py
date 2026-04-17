"""Model registry for ThroughputBencher benchmarks.

Each model entry defines the timm model name, display metadata, and whether
it supports SMP U-Net segmentation (requires hierarchical multi-scale features).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    timm_name: str
    display_name: str
    family: str
    arch_type: str  # "cnn", "vit", "hybrid"
    color: str
    supports_segmentation: bool = True

    @property
    def smp_encoder_name(self) -> str:
        return f"tu-{self.timm_name}"


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
}

MODEL_REGISTRY: list[ModelConfig] = [
    # ── CNNs ──────────────────────────────────────────────────────────────
    # ResNet family
    ModelConfig("resnet18", "ResNet-18", "ResNet", "cnn", FAMILY_COLORS["ResNet"]),
    ModelConfig("resnet50", "ResNet-50", "ResNet", "cnn", FAMILY_COLORS["ResNet"]),
    ModelConfig("resnet101", "ResNet-101", "ResNet", "cnn", FAMILY_COLORS["ResNet"]),
    ModelConfig("resnet152", "ResNet-152", "ResNet", "cnn", FAMILY_COLORS["ResNet"]),
    # EfficientNet family
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
    # ConvNeXt family
    ModelConfig("convnext_tiny", "ConvNeXt-T", "ConvNeXt", "cnn", FAMILY_COLORS["ConvNeXt"]),
    ModelConfig("convnext_small", "ConvNeXt-S", "ConvNeXt", "cnn", FAMILY_COLORS["ConvNeXt"]),
    ModelConfig("convnext_base", "ConvNeXt-B", "ConvNeXt", "cnn", FAMILY_COLORS["ConvNeXt"]),
    ModelConfig("convnext_large", "ConvNeXt-L", "ConvNeXt", "cnn", FAMILY_COLORS["ConvNeXt"]),
    # MobileNet family
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
    # RegNet family
    ModelConfig("regnety_004", "RegNetY-400MF", "RegNet", "cnn", FAMILY_COLORS["RegNet"]),
    ModelConfig("regnety_040", "RegNetY-4GF", "RegNet", "cnn", FAMILY_COLORS["RegNet"]),
    # ── Vision Transformers ───────────────────────────────────────────────
    # Plain ViTs do NOT produce hierarchical multi-scale features, so SMP
    # U-Net (which expects an encoder with multiple resolution stages) cannot
    # use them.  Segmentation is skipped for these models.
    ModelConfig(
        "vit_tiny_patch16_224",
        "ViT-Ti/16",
        "ViT",
        "vit",
        FAMILY_COLORS["ViT"],
        supports_segmentation=False,
    ),
    ModelConfig(
        "vit_small_patch16_224",
        "ViT-S/16",
        "ViT",
        "vit",
        FAMILY_COLORS["ViT"],
        supports_segmentation=False,
    ),
    ModelConfig(
        "vit_base_patch16_224",
        "ViT-B/16",
        "ViT",
        "vit",
        FAMILY_COLORS["ViT"],
        supports_segmentation=False,
    ),
    ModelConfig(
        "vit_large_patch16_224",
        "ViT-L/16",
        "ViT",
        "vit",
        FAMILY_COLORS["ViT"],
        supports_segmentation=False,
    ),
    # DeiT family
    ModelConfig(
        "deit3_small_patch16_224",
        "DeiT3-S/16",
        "DeiT",
        "vit",
        FAMILY_COLORS["DeiT"],
        supports_segmentation=False,
    ),
    ModelConfig(
        "deit3_base_patch16_224",
        "DeiT3-B/16",
        "DeiT",
        "vit",
        FAMILY_COLORS["DeiT"],
        supports_segmentation=False,
    ),
    # Swin Transformer family (hierarchical — segmentation works)
    ModelConfig("swin_tiny_patch4_window7_224", "Swin-T", "Swin", "vit", FAMILY_COLORS["Swin"]),
    ModelConfig("swin_small_patch4_window7_224", "Swin-S", "Swin", "vit", FAMILY_COLORS["Swin"]),
    ModelConfig("swin_base_patch4_window7_224", "Swin-B", "Swin", "vit", FAMILY_COLORS["Swin"]),
    ModelConfig("swin_large_patch4_window7_224", "Swin-L", "Swin", "vit", FAMILY_COLORS["Swin"]),
    # BEiT family
    ModelConfig(
        "beit_base_patch16_224",
        "BEiT-B/16",
        "BEiT",
        "vit",
        FAMILY_COLORS["BEiT"],
        supports_segmentation=False,
    ),
    ModelConfig(
        "beit_large_patch16_224",
        "BEiT-L/16",
        "BEiT",
        "vit",
        FAMILY_COLORS["BEiT"],
        supports_segmentation=False,
    ),
    # ── Hybrids ───────────────────────────────────────────────────────────
    ModelConfig("coatnet_0_224", "CoAtNet-0", "CoAtNet", "hybrid", FAMILY_COLORS["CoAtNet"]),
    ModelConfig("coatnet_2_224", "CoAtNet-2", "CoAtNet", "hybrid", FAMILY_COLORS["CoAtNet"]),
]


def get_models(names: list[str] | None = None) -> list[ModelConfig]:
    """Return models filtered by timm name. If names is None, return all."""
    if names is None:
        return list(MODEL_REGISTRY)
    name_set = set(names)
    return [m for m in MODEL_REGISTRY if m.timm_name in name_set]
