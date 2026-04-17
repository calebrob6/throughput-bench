#!/usr/bin/env python3
"""Simple timm ResNet-18 throughput benchmark.

Runs forward passes on synthetic 3x224x224 inputs using a random dataset and
reports images/sec over ~60 seconds of timed runtime.

Example:
    python benchmark_resnet18.py --device cuda:0
    python benchmark_resnet18.py --device cuda:1 --batch-size 512 --num-workers 8
"""

import argparse
import time

import timm
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from data import create_dataloader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--runtime-seconds", type=float, default=60.0)
    parser.add_argument("--warmup-steps", type=int, default=10)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    device = torch.device(args.device)

    model = timm.create_model("resnet18", pretrained=False, num_classes=10)
    model.eval().to(device)

    dataloader = create_dataloader(
        task="classification",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        length=100_000,
    )

    # Warmup
    with torch.inference_mode():
        data_iter = iter(dataloader)
        for _ in range(args.warmup_steps):
            images, _ = next(data_iter)
            images = images.to(device, non_blocking=True)
            _ = model(images)
        torch.cuda.synchronize(device)

    # Timed run
    total_images = 0
    start_time = time.perf_counter()
    pbar = tqdm(total=int(args.runtime_seconds), unit="s", desc="Benchmarking",
                bar_format="{desc}: {elapsed} | {postfix}")
    last_update = start_time
    with torch.inference_mode():
        data_iter = iter(dataloader)
        while True:
            now = time.perf_counter()
            if now - start_time >= args.runtime_seconds:
                break

            try:
                images, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                images, _ = next(data_iter)

            images = images.to(device, non_blocking=True)
            _ = model(images)
            total_images += images.shape[0]

            if now - last_update >= 0.5:
                elapsed_so_far = now - start_time
                pbar.set_postfix_str(
                    f"{total_images / elapsed_so_far:.0f} img/s | "
                    f"{total_images:,} images"
                )
                last_update = now

    elapsed = time.perf_counter() - start_time
    images_per_second = total_images / elapsed
    pbar.set_postfix_str(
        f"{images_per_second:.0f} img/s | {total_images:,} images (final)"
    )
    pbar.close()

    print(f"device: {args.device}")
    print(f"batch_size: {args.batch_size}")
    print(f"runtime_seconds: {elapsed:.2f}")
    print(f"total_images: {total_images}")
    print(f"images_per_second: {images_per_second:.2f}")


if __name__ == "__main__":
    main()