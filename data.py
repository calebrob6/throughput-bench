"""Random patch dataset and dataloader for benchmarking.

Generates synthetic 3-channel 224×224 patches with dummy labels so that
benchmarks measure pure compute throughput without disk I/O.
"""

import torch
from torch.utils.data import Dataset, DataLoader


class RandomPatchDataset(Dataset):
    """Dataset of random 3×224×224 tensors for throughput benchmarking.

    For classification returns (image, label).
    For segmentation returns (image, mask).
    """

    def __init__(
        self,
        length: int = 10_000,
        channels: int = 3,
        size: int = 224,
        num_classes: int = 10,
        task: str = "classification",
    ):
        self.length = length
        self.channels = channels
        self.size = size
        self.num_classes = num_classes
        self.task = task

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        image = torch.ones(self.channels, self.size, self.size)
        if self.task == "classification":
            label = torch.randint(0, self.num_classes, (1,)).item()
            return image, label
        else:
            mask = torch.randint(0, self.num_classes, (self.size, self.size))
            return image, mask


def create_dataloader(
    task: str = "classification",
    batch_size: int = 32,
    num_workers: int = 4,
    prefetch_factor: int | None = 2,
    length: int = 10_000,
    **kwargs,
) -> DataLoader:
    """Create a DataLoader with random patches."""
    dataset = RandomPatchDataset(length=length, task=task)
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        **kwargs,
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **loader_kwargs)
