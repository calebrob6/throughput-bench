"""Synthetic dataset and DataLoader for throughput benchmarking.

Generates synthetic patches with dummy labels. Supports three data modes:
  ones     — constant tensor of ones (zero variance, legacy DataLoader default)
  randn    — standard-normal noise (legacy pre-allocated default)
  spectral — per-band normal samples drawn from approximate S2/Landsat statistics
             (produces realistic attention diversity in ViT-based geospatial FMs)
"""

import torch
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Per-band spectral statistics
# ---------------------------------------------------------------------------

# Sentinel-2 L2A surface reflectance, 0-1 scale.
# Order: B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12
_S2_MEANS = [0.080, 0.083, 0.102, 0.085, 0.138, 0.199, 0.228, 0.237, 0.244, 0.248, 0.010, 0.147, 0.082]
_S2_STDS  = [0.038, 0.043, 0.051, 0.053, 0.064, 0.085, 0.096, 0.103, 0.106, 0.102, 0.012, 0.091, 0.053]

# Landsat 8/9 OLI surface reflectance.
# Order: B1 (coastal), B2 (blue), B3 (green), B4 (red), B5 (NIR), B6 (SWIR1), B7 (SWIR2)
_L8_MEANS = [0.072, 0.074, 0.093, 0.072, 0.216, 0.140, 0.081]
_L8_STDS  = [0.030, 0.033, 0.041, 0.042, 0.098, 0.082, 0.050]

# Known-channel mappings.  For anything else we cycle through S2 stats.
_CHANNEL_STATS: dict[int, tuple[list[float], list[float]]] = {
    3:  (_S2_MEANS[1:4],  _S2_STDS[1:4]),   # S2 RGB  (B02, B03, B04)
    4:  (_S2_MEANS[1:4] + [_S2_MEANS[7]], _S2_STDS[1:4] + [_S2_STDS[7]]),  # RGBNIR
    6:  (_L8_MEANS[1:],   _L8_STDS[1:]),     # Landsat B2-B7
    7:  (_L8_MEANS,       _L8_STDS),          # Landsat B1-B7
    12: (_S2_MEANS[:10] + _S2_MEANS[11:], _S2_STDS[:10] + _S2_STDS[11:]),  # S2 minus B10
    13: (_S2_MEANS,       _S2_STDS),          # All 13 S2 bands
}


def _band_stats(channels: int) -> tuple[list[float], list[float]]:
    """Return (means, stds) for *channels* bands, cycling through S2 if unknown."""
    if channels in _CHANNEL_STATS:
        return _CHANNEL_STATS[channels]
    # Cycle through full S2 stats for arbitrary channel counts
    means = [_S2_MEANS[i % 13] for i in range(channels)]
    stds  = [_S2_STDS[i % 13]  for i in range(channels)]
    return means, stds


def make_spectral_batch(
    batch_size: int,
    channels: int,
    size: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return a (B, C, H, W) tensor sampled from per-band S2/Landsat statistics."""
    means, stds = _band_stats(channels)
    # Shape (1, C, 1, 1) for broadcasting
    mu  = torch.tensor(means, dtype=torch.float32).view(1, channels, 1, 1)
    sig = torch.tensor(stds,  dtype=torch.float32).view(1, channels, 1, 1)
    noise = torch.randn(batch_size, channels, size, size)
    batch = mu + sig * noise
    if device is not None:
        batch = batch.to(device)
    return batch


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class RandomPatchDataset(Dataset):
    """Synthetic patches with dummy labels for throughput benchmarking."""

    def __init__(
        self,
        length: int = 10_000,
        channels: int = 3,
        size: int = 224,
        num_classes: int = 10,
        data_mode: str = "ones",
    ):
        self.length = length
        self.channels = channels
        self.size = size
        self.num_classes = num_classes
        self.data_mode = data_mode
        if data_mode == "spectral":
            means, stds = _band_stats(channels)
            self._mu  = torch.tensor(means, dtype=torch.float32).view(channels, 1, 1)
            self._sig = torch.tensor(stds,  dtype=torch.float32).view(channels, 1, 1)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, _: int):
        if self.data_mode == "spectral":
            image = self._mu + self._sig * torch.randn(self.channels, self.size, self.size)
        elif self.data_mode == "randn":
            image = torch.randn(self.channels, self.size, self.size)
        else:
            image = torch.ones(self.channels, self.size, self.size)
        label = torch.randint(0, self.num_classes, (1,)).item()
        return image, label


def create_dataloader(
    batch_size: int = 32,
    num_workers: int = 4,
    prefetch_factor: int | None = 2,
    length: int = 10_000,
    channels: int = 3,
    size: int = 224,
    data_mode: str = "ones",
    **kwargs,
) -> DataLoader:
    """Create a DataLoader with synthetic patches."""
    dataset = RandomPatchDataset(length=length, channels=channels, size=size, data_mode=data_mode)
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        **kwargs,
    )
    if num_workers > 0:
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **loader_kwargs)
