#!/usr/bin/env python3
"""Sweep DataLoader IPC impact for one fixed model across batch sizes.

The dataset returns the same pre-generated random CPU tensor for every sample.
That keeps ``__getitem__`` close to a best-case path and makes the measured
DataLoader cost mostly batching, worker IPC, pinning, and main-process dequeue
overhead rather than image decoding or augmentation work.

Example:
    python benchmark_dataloader_ipc.py --device 0
    python benchmark_dataloader_ipc.py --device 0 --batch-sizes 32 64 128 256 512
"""

import argparse
import csv
import gc
import time
from pathlib import Path
from typing import Any

import timm
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

CSV_COLUMNS = [
    "model",
    "batch_size",
    "batch_mb",
    "input_channels",
    "input_size",
    "num_workers",
    "prefetch_factor",
    "pin_memory",
    "warmup_steps",
    "iterations",
    "compute_mean_ms",
    "compute_p50_ms",
    "compute_p95_ms",
    "compute_img_s",
    "fetch_only_mean_ms",
    "fetch_only_p50_ms",
    "fetch_only_p95_ms",
    "fetch_only_img_s",
    "e2e_mean_ms",
    "e2e_p50_ms",
    "e2e_p95_ms",
    "e2e_img_s",
    "e2e_fetch_mean_ms",
    "e2e_gpu_mean_ms",
    "overhead_vs_compute_ms",
    "slowdown_vs_compute",
    "fetch_only_pct_of_compute",
    "gpu_name",
    "pytorch_version",
    "cuda_version",
    "timestamp",
]


class CachedRandomTensorDataset(Dataset):
    """Best-case dummy dataset that returns a cached random tensor."""

    def __init__(
        self,
        length: int,
        channels: int,
        size: int,
        num_classes: int,
    ):
        self.length = length
        self.image = torch.randn(channels, size, size)
        self.label = torch.randint(0, num_classes, (1,)).item()

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.image, self.label


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_seconds(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
    mean_seconds = sum(values) / len(values)
    return {
        "mean_ms": mean_seconds * 1_000.0,
        "p50_ms": percentile(values, 0.50) * 1_000.0,
        "p95_ms": percentile(values, 0.95) * 1_000.0,
    }


def make_dataloader(
    *,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
    iterations: int,
    warmup_steps: int,
    channels: int,
    size: int,
    num_classes: int,
) -> DataLoader:
    batches_needed = iterations + warmup_steps + max(num_workers * prefetch_factor, 0) + 8
    dataset = CachedRandomTensorDataset(
        length=batch_size * batches_needed,
        channels=channels,
        size=size,
        num_classes=num_classes,
    )
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": True,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["persistent_workers"] = True
    return DataLoader(dataset, **kwargs)


def next_batch(data_iter: Any, dataloader: DataLoader) -> tuple[Any, Any]:
    try:
        return next(data_iter), data_iter
    except StopIteration:
        data_iter = iter(dataloader)
        return next(data_iter), data_iter


def shutdown_iterator(data_iter: Any) -> None:
    shutdown_workers = getattr(data_iter, "_shutdown_workers", None)
    if shutdown_workers is not None:
        shutdown_workers()


def benchmark_compute_only(
    model: torch.nn.Module,
    *,
    batch_size: int,
    channels: int,
    size: int,
    device: torch.device,
    warmup_steps: int,
    iterations: int,
) -> dict[str, float]:
    images = torch.randn(batch_size, channels, size, size, device=device)
    times: list[float] = []

    with torch.inference_mode():
        for _ in range(warmup_steps):
            _ = model(images)
        torch.cuda.synchronize(device)

        for _ in range(iterations):
            start = time.perf_counter()
            _ = model(images)
            torch.cuda.synchronize(device)
            times.append(time.perf_counter() - start)

    stats = summarize_seconds(times)
    mean_seconds = stats["mean_ms"] / 1_000.0
    stats["img_s"] = batch_size / mean_seconds if mean_seconds > 0 else 0.0
    return stats


def benchmark_fetch_only(
    *,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
    iterations: int,
    warmup_steps: int,
    channels: int,
    size: int,
    num_classes: int,
) -> dict[str, float]:
    dataloader = make_dataloader(
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        iterations=iterations,
        warmup_steps=warmup_steps,
        channels=channels,
        size=size,
        num_classes=num_classes,
    )
    data_iter = iter(dataloader)
    times: list[float] = []

    try:
        for _ in range(warmup_steps):
            _, data_iter = next_batch(data_iter, dataloader)

        for _ in range(iterations):
            start = time.perf_counter()
            _, data_iter = next_batch(data_iter, dataloader)
            times.append(time.perf_counter() - start)
    finally:
        shutdown_iterator(data_iter)
        del data_iter, dataloader
        gc.collect()

    stats = summarize_seconds(times)
    mean_seconds = stats["mean_ms"] / 1_000.0
    stats["img_s"] = batch_size / mean_seconds if mean_seconds > 0 else 0.0
    return stats


def benchmark_e2e(
    model: torch.nn.Module,
    *,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
    iterations: int,
    warmup_steps: int,
    channels: int,
    size: int,
    num_classes: int,
    device: torch.device,
) -> dict[str, float]:
    dataloader = make_dataloader(
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        iterations=iterations,
        warmup_steps=warmup_steps,
        channels=channels,
        size=size,
        num_classes=num_classes,
    )
    data_iter = iter(dataloader)
    total_times: list[float] = []
    fetch_times: list[float] = []
    gpu_times: list[float] = []

    try:
        with torch.inference_mode():
            for _ in range(warmup_steps):
                batch, data_iter = next_batch(data_iter, dataloader)
                images, _ = batch
                images = images.to(device, non_blocking=pin_memory)
                _ = model(images)
            torch.cuda.synchronize(device)

            for _ in range(iterations):
                total_start = time.perf_counter()

                fetch_start = time.perf_counter()
                batch, data_iter = next_batch(data_iter, dataloader)
                fetch_elapsed = time.perf_counter() - fetch_start

                gpu_start = time.perf_counter()
                images, _ = batch
                images = images.to(device, non_blocking=pin_memory)
                _ = model(images)
                torch.cuda.synchronize(device)
                gpu_elapsed = time.perf_counter() - gpu_start

                total_times.append(time.perf_counter() - total_start)
                fetch_times.append(fetch_elapsed)
                gpu_times.append(gpu_elapsed)
    finally:
        shutdown_iterator(data_iter)
        del data_iter, dataloader
        gc.collect()

    stats = summarize_seconds(total_times)
    fetch_stats = summarize_seconds(fetch_times)
    gpu_stats = summarize_seconds(gpu_times)
    mean_seconds = stats["mean_ms"] / 1_000.0
    stats["img_s"] = batch_size / mean_seconds if mean_seconds > 0 else 0.0
    stats["fetch_mean_ms"] = fetch_stats["mean_ms"]
    stats["gpu_mean_ms"] = gpu_stats["mean_ms"]
    return stats


def batch_size_mb(batch_size: int, channels: int, size: int) -> float:
    element_size = torch.tensor([], dtype=torch.float32).element_size()
    bytes_per_batch = batch_size * channels * size * size * element_size
    return bytes_per_batch / 1_000_000.0


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure fixed-model DataLoader IPC impact across batch sizes."
    )
    parser.add_argument("--device", type=nonnegative_int, default=0, help="CUDA device index")
    parser.add_argument("--model", type=str, default="resnet18", help="Fixed timm model name")
    parser.add_argument(
        "--batch-sizes",
        type=positive_int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
        help="Batch sizes to sweep",
    )
    parser.add_argument("--num-workers", type=nonnegative_int, default=8)
    parser.add_argument("--prefetch-factor", type=positive_int, default=2)
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use DataLoader pin_memory and non-blocking GPU copies",
    )
    parser.add_argument("--iterations", type=positive_int, default=50)
    parser.add_argument("--warmup-steps", type=nonnegative_int, default=10)
    parser.add_argument("--input-channels", type=positive_int, default=3)
    parser.add_argument("--input-size", type=positive_int, default=224)
    parser.add_argument("--num-classes", type=positive_int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/dataloader_ipc_sweep.csv"),
        help="CSV output path",
    )
    parser.add_argument("--append", action="store_true", help="Append to an existing CSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    torch.set_float32_matmul_precision("high")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)

    model = timm.create_model(
        args.model,
        pretrained=False,
        num_classes=args.num_classes,
        in_chans=args.input_channels,
    )
    model.eval().to(device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.append or not args.output.exists()
    mode = "a" if args.append else "w"

    with args.output.open(mode, newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()

        progress = tqdm(args.batch_sizes, desc="Sweeping batch sizes", unit="batch")
        for batch_size in progress:
            progress.set_postfix_str(f"batch={batch_size}")
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

            compute = benchmark_compute_only(
                model,
                batch_size=batch_size,
                channels=args.input_channels,
                size=args.input_size,
                device=device,
                warmup_steps=args.warmup_steps,
                iterations=args.iterations,
            )
            fetch_only = benchmark_fetch_only(
                batch_size=batch_size,
                num_workers=args.num_workers,
                prefetch_factor=args.prefetch_factor,
                pin_memory=args.pin_memory,
                iterations=args.iterations,
                warmup_steps=args.warmup_steps,
                channels=args.input_channels,
                size=args.input_size,
                num_classes=args.num_classes,
            )
            e2e = benchmark_e2e(
                model,
                batch_size=batch_size,
                num_workers=args.num_workers,
                prefetch_factor=args.prefetch_factor,
                pin_memory=args.pin_memory,
                iterations=args.iterations,
                warmup_steps=args.warmup_steps,
                channels=args.input_channels,
                size=args.input_size,
                num_classes=args.num_classes,
                device=device,
            )

            overhead_ms = e2e["mean_ms"] - compute["mean_ms"]
            slowdown = e2e["mean_ms"] / compute["mean_ms"] if compute["mean_ms"] > 0 else 0.0
            fetch_pct = (
                100.0 * fetch_only["mean_ms"] / compute["mean_ms"]
                if compute["mean_ms"] > 0
                else 0.0
            )

            row = {
                "model": args.model,
                "batch_size": batch_size,
                "batch_mb": batch_size_mb(batch_size, args.input_channels, args.input_size),
                "input_channels": args.input_channels,
                "input_size": args.input_size,
                "num_workers": args.num_workers,
                "prefetch_factor": args.prefetch_factor if args.num_workers > 0 else "",
                "pin_memory": args.pin_memory,
                "warmup_steps": args.warmup_steps,
                "iterations": args.iterations,
                "compute_mean_ms": compute["mean_ms"],
                "compute_p50_ms": compute["p50_ms"],
                "compute_p95_ms": compute["p95_ms"],
                "compute_img_s": compute["img_s"],
                "fetch_only_mean_ms": fetch_only["mean_ms"],
                "fetch_only_p50_ms": fetch_only["p50_ms"],
                "fetch_only_p95_ms": fetch_only["p95_ms"],
                "fetch_only_img_s": fetch_only["img_s"],
                "e2e_mean_ms": e2e["mean_ms"],
                "e2e_p50_ms": e2e["p50_ms"],
                "e2e_p95_ms": e2e["p95_ms"],
                "e2e_img_s": e2e["img_s"],
                "e2e_fetch_mean_ms": e2e["fetch_mean_ms"],
                "e2e_gpu_mean_ms": e2e["gpu_mean_ms"],
                "overhead_vs_compute_ms": overhead_ms,
                "slowdown_vs_compute": slowdown,
                "fetch_only_pct_of_compute": fetch_pct,
                "gpu_name": torch.cuda.get_device_name(device),
                "pytorch_version": torch.__version__,
                "cuda_version": torch.version.cuda or "N/A",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            writer.writerow(row)
            csv_file.flush()

            print(
                f"batch={batch_size:<4} "
                f"compute={compute['mean_ms']:.2f}ms "
                f"fetch={fetch_only['mean_ms']:.2f}ms "
                f"e2e={e2e['mean_ms']:.2f}ms "
                f"slowdown={slowdown:.2f}x"
            )

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
