"""Synthetic dataset and DataLoader for throughput benchmarking.

Generates constant patches with dummy labels. Used by
benchmark.py when running with the DataLoader path.
"""

import torch
from torch.utils.data import DataLoader, Dataset


class RandomPatchDataset(Dataset):
    """Dataset of constant tensors with dummy labels for throughput benchmarking."""

    def __init__(
        self,
        length: int = 10_000,
        channels: int = 3,
        size: int = 224,
        num_classes: int = 10,
    ):
        self.length = length
        self.channels = channels
        self.size = size
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
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
    **kwargs,
) -> DataLoader:
    """Create a DataLoader with random patches."""
    dataset = RandomPatchDataset(length=length, channels=channels, size=size)
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
