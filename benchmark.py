#!/usr/bin/env python3
"""GeoSpeedy — rigorous throughput benchmarking for geospatial model backbones.

Measures inference throughput (images/sec) for classification and segmentation
across many model architectures, precision modes, and hardware configurations.

Usage examples:
    # Full GPU benchmark on GPU 0
    python benchmark.py --gpu-id 0

    # Quick test with one model
    python benchmark.py --gpu-id 0 --models resnet50 --batch-sizes 32

    # CPU benchmark with varying thread counts
    python benchmark.py --device cpu --num-threads 1 4 8

    # Only classification, AMP precision
    python benchmark.py --gpu-id 0 --tasks classification --precisions amp
"""

import argparse
import csv
import gc
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import timm

try:
    import segmentation_models_pytorch as smp
except ImportError:
    smp = None

from data import create_dataloader
from models import MODEL_REGISTRY, ModelConfig, get_models

# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "model_name",
    "display_name",
    "model_family",
    "model_type",
    "task",
    "precision",
    "compiled",
    "compile_mode",
    "device",
    "gpu_name",
    "batch_size",
    "num_threads",
    "throughput_mean",
    "throughput_std",
    "throughput_median",
    "throughput_min",
    "throughput_max",
    "pixels_per_sec",
    "latency_mean_ms",
    "latency_std_ms",
    "params_M",
    "macs_G",
    "peak_memory_mb",
    "pytorch_version",
    "timestamp",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def count_params(model: nn.Module) -> float:
    """Return parameter count in millions."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def estimate_macs(model: nn.Module, input_shape: tuple = (1, 3, 224, 224),
                  device: str = "cpu") -> float:
    """Estimate MACs in GFLOPs using PyTorch's flop counter."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
        inp = torch.randn(*input_shape, device=device)
        with FlopCounterMode(display=False) as fcm:
            model(inp)
        total_flops = fcm.get_total_flops()
        return total_flops / 1e9  # GFLOPs
    except Exception:
        return -1.0


def get_gpu_name(gpu_id: int = 0) -> str:
    """Return GPU name string."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(gpu_id)
    return "N/A"


def check_gpu_free(gpu_id: int) -> bool:
    """Check that the target GPU has no other processes running."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        gpu_uuids_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        target_uuid = None
        for line in gpu_uuids_result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2 and parts[0] == str(gpu_id):
                target_uuid = parts[1]
                break
        if target_uuid is None:
            return True  # can't determine, assume free

        our_pid = str(os.getpid())
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2:
                pid, uuid = parts
                if uuid == target_uuid and pid != our_pid:
                    return False
        return True
    except Exception:
        return True  # can't check, proceed


def gpu_cleanup():
    """Aggressively free GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# Model creation
# ---------------------------------------------------------------------------


def create_model_for_task(
    config: ModelConfig, task: str, device: torch.device
) -> nn.Module | None:
    """Instantiate a model for the given task. Returns None if unsupported."""
    if task == "classification":
        model = timm.create_model(config.timm_name, pretrained=False,
                                  num_classes=10)
    elif task == "segmentation":
        if not config.supports_segmentation:
            return None
        if smp is None:
            print(f"    ⚠ segmentation_models_pytorch not installed, "
                  f"skipping segmentation for {config.display_name}")
            return None
        model = smp.Unet(
            encoder_name=config.smp_encoder_name,
            encoder_weights=None,
            in_channels=3,
            classes=10,
        )
    else:
        raise ValueError(f"Unknown task: {task}")

    model = model.to(device)
    model.eval()
    return model


def apply_precision(model: nn.Module, precision: str,
                    device: torch.device) -> nn.Module:
    """Apply precision mode to model. AMP is handled at inference time."""
    if precision == "fp16":
        model = model.half()
    return model


def apply_compile(model: nn.Module, compile_mode: str) -> tuple[nn.Module, bool]:
    """Apply torch.compile if requested. Returns (model, success)."""
    if compile_mode == "none":
        return model, True
    try:
        model = torch.compile(model, mode=compile_mode)
        return model, True
    except Exception as e:
        print(f"    ⚠ torch.compile({compile_mode}) failed: {e}")
        return model, False


# ---------------------------------------------------------------------------
# Benchmarking core
# ---------------------------------------------------------------------------


def benchmark_gpu(
    model: nn.Module,
    dataloader,
    precision: str,
    device: torch.device,
    num_warmup: int = 20,
    min_timed_seconds: float = 10.0,
) -> dict:
    """Benchmark on GPU using cuda Events for precise timing.

    Runs warmup iterations, then times iterations for at least
    ``min_timed_seconds`` to get stable statistics.
    """
    use_amp = precision == "amp"
    use_fp16_input = precision == "fp16"

    # --- warmup ---
    data_iter = iter(dataloader)
    for _ in range(num_warmup):
        try:
            images, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            images, _ = next(data_iter)
        images = images.to(device, non_blocking=True)
        if use_fp16_input:
            images = images.half()
        with torch.no_grad():
            if use_amp:
                with torch.amp.autocast("cuda"):
                    _ = model(images)
            else:
                _ = model(images)
    torch.cuda.synchronize()

    # Reset peak memory AFTER warmup so we measure steady-state only
    torch.cuda.reset_peak_memory_stats()

    # --- timed iterations ---
    batch_size = dataloader.batch_size
    timings_ms: list[float] = []
    total_elapsed = 0.0
    data_iter = iter(dataloader)

    while total_elapsed < min_timed_seconds:
        try:
            images, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            images, _ = next(data_iter)

        images = images.to(device, non_blocking=True)
        if use_fp16_input:
            images = images.half()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        with torch.no_grad():
            if use_amp:
                with torch.amp.autocast("cuda"):
                    _ = model(images)
            else:
                _ = model(images)
        end_event.record()
        torch.cuda.synchronize()

        elapsed_ms = start_event.elapsed_time(end_event)
        timings_ms.append(elapsed_ms)
        total_elapsed += elapsed_ms / 1000.0

    timings = np.array(timings_ms)

    # Aggregate throughput = total_images / total_time (not mean of rates)
    total_images = batch_size * len(timings)
    total_time_s = timings.sum() / 1000.0
    aggregate_throughput = total_images / total_time_s

    # Per-iteration rates for reporting spread
    per_iter_rates = batch_size / (timings / 1000.0)

    peak_mem = torch.cuda.max_memory_allocated() / 1e6  # MB

    return {
        "throughput_mean": float(aggregate_throughput),
        "throughput_std": float(np.std(per_iter_rates)),
        "throughput_median": float(np.median(per_iter_rates)),
        "throughput_min": float(np.min(per_iter_rates)),
        "throughput_max": float(np.max(per_iter_rates)),
        "latency_mean_ms": float(np.mean(timings)),
        "latency_std_ms": float(np.std(timings)),
        "peak_memory_mb": float(peak_mem),
        "num_iterations": len(timings),
    }


def benchmark_cpu(
    model: nn.Module,
    dataloader,
    num_warmup: int = 5,
    min_timed_seconds: float = 10.0,
) -> dict:
    """Benchmark on CPU using time.perf_counter."""
    # --- warmup ---
    data_iter = iter(dataloader)
    for _ in range(num_warmup):
        try:
            images, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            images, _ = next(data_iter)
        with torch.no_grad():
            _ = model(images)

    # --- timed ---
    batch_size = dataloader.batch_size
    timings_ms: list[float] = []
    total_elapsed = 0.0
    data_iter = iter(dataloader)

    while total_elapsed < min_timed_seconds:
        try:
            images, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            images, _ = next(data_iter)

        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(images)
        t1 = time.perf_counter()

        elapsed_ms = (t1 - t0) * 1000.0
        timings_ms.append(elapsed_ms)
        total_elapsed += elapsed_ms / 1000.0

    timings = np.array(timings_ms)

    total_images = batch_size * len(timings)
    total_time_s = timings.sum() / 1000.0
    aggregate_throughput = total_images / total_time_s
    per_iter_rates = batch_size / (timings / 1000.0)

    return {
        "throughput_mean": float(aggregate_throughput),
        "throughput_std": float(np.std(per_iter_rates)),
        "throughput_median": float(np.median(per_iter_rates)),
        "throughput_min": float(np.min(per_iter_rates)),
        "throughput_max": float(np.max(per_iter_rates)),
        "latency_mean_ms": float(np.mean(timings)),
        "latency_std_ms": float(np.std(timings)),
        "peak_memory_mb": 0.0,
        "num_iterations": len(timings),
    }


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------


def run_benchmark(args):
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine device
    if args.device == "cuda":
        gpu_id = args.gpu_id
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        # After setting CUDA_VISIBLE_DEVICES, device 0 maps to the chosen GPU
        device = torch.device("cuda:0")
        gpu_name = get_gpu_name(0)
        print(f"🖥  Device: GPU {gpu_id} ({gpu_name})")

        if not check_gpu_free(gpu_id):
            print(f"⚠  WARNING: Other processes detected on GPU {gpu_id}. "
                  f"Results may be unreliable.")
    else:
        device = torch.device("cpu")
        gpu_name = "N/A"
        print(f"🖥  Device: CPU")

    # Resolve model list
    model_configs = get_models(args.models if args.models else None)
    print(f"📋 Models: {len(model_configs)}")
    print(f"📋 Tasks: {args.tasks}")
    print(f"📋 Precisions: {args.precisions}")
    print(f"📋 Compile modes: {args.compile_modes}")
    print(f"📋 Batch sizes: {args.batch_sizes}")

    # Open CSV writer (append mode so we can resume)
    file_exists = output_path.exists() and output_path.stat().st_size > 0
    csv_file = open(output_path, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
    if not file_exists:
        writer.writeheader()

    total_configs = 0
    completed = 0

    # Count total configs for progress
    for mc in model_configs:
        for task in args.tasks:
            if task == "segmentation" and not mc.supports_segmentation:
                continue
            for prec in args.precisions:
                if args.device == "cpu" and prec in ("fp16", "amp"):
                    continue
                for cm in args.compile_modes:
                    for bs in args.batch_sizes:
                        total_configs += 1

    if args.device == "cpu" and args.num_threads:
        total_configs *= len(args.num_threads)

    print(f"\n🔢 Total benchmark configurations: {total_configs}\n")

    for mc in model_configs:
        print(f"\n{'='*70}")
        print(f"  {mc.display_name} ({mc.timm_name}) — {mc.arch_type}")
        print(f"{'='*70}")

        # Compute MACs once per model (on CPU to avoid GPU memory)
        macs_g = -1.0
        params_m = -1.0
        try:
            tmp_model = timm.create_model(mc.timm_name, pretrained=False,
                                          num_classes=10)
            tmp_model.eval()
            params_m = count_params(tmp_model)
            macs_g = estimate_macs(tmp_model, device="cpu")
            del tmp_model
            gc.collect()
        except Exception as e:
            print(f"  ⚠ Could not compute MACs: {e}")

        for task in args.tasks:
            if task == "segmentation" and not mc.supports_segmentation:
                print(f"  ⏭ Skipping segmentation (not supported by SMP U-Net)")
                continue

            for prec in args.precisions:
                if args.device == "cpu" and prec in ("fp16", "amp"):
                    continue

                for cm in args.compile_modes:
                    for bs in args.batch_sizes:
                        thread_counts = (
                            args.num_threads
                            if args.device == "cpu" and args.num_threads
                            else [0]
                        )

                        for nt in thread_counts:
                            completed += 1
                            label = (f"  [{completed}/{total_configs}] "
                                     f"{task} | {prec} | compile={cm} | "
                                     f"bs={bs}")
                            if nt > 0:
                                label += f" | threads={nt}"
                            print(label, end=" ... ", flush=True)

                            if args.device == "cpu" and nt > 0:
                                torch.set_num_threads(nt)

                            gpu_cleanup()

                            try:
                                model = create_model_for_task(mc, task, device)
                                if model is None:
                                    print("SKIP")
                                    continue

                                model = apply_precision(model, prec, device)
                                model, compile_ok = apply_compile(model, cm)
                                actual_compile_mode = cm if compile_ok else "none"
                                actual_compiled = compile_ok and cm != "none"

                                # Compute seg-specific MACs if needed
                                task_macs = macs_g
                                task_params = params_m
                                if task == "segmentation":
                                    try:
                                        task_params = count_params(model)
                                        task_macs = estimate_macs(
                                            model,
                                            input_shape=(1, 3, 224, 224),
                                            device=str(device) if args.device == "cpu" else "cpu",
                                        )
                                    except Exception:
                                        pass

                                # Force num_workers=0 on CPU to avoid
                                # worker threads competing for cores
                                dl_workers = (0 if args.device == "cpu"
                                              else args.num_workers)
                                dl = create_dataloader(
                                    task=task,
                                    batch_size=bs,
                                    num_workers=dl_workers,
                                    length=max(
                                        bs * 500,
                                        10_000,
                                    ),
                                )

                                if args.device == "cuda":
                                    torch.cuda.reset_peak_memory_stats()
                                    stats = benchmark_gpu(
                                        model, dl, prec, device,
                                        num_warmup=args.warmup,
                                        min_timed_seconds=args.timed_seconds,
                                    )
                                else:
                                    stats = benchmark_cpu(
                                        model, dl,
                                        num_warmup=max(2, args.warmup // 5),
                                        min_timed_seconds=args.timed_seconds,
                                    )

                                pixels_per_sec = stats["throughput_mean"] * 224 * 224

                                row = {
                                    "model_name": mc.timm_name,
                                    "display_name": mc.display_name,
                                    "model_family": mc.family,
                                    "model_type": mc.arch_type,
                                    "task": task,
                                    "precision": prec,
                                    "compiled": actual_compiled,
                                    "compile_mode": actual_compile_mode,
                                    "device": args.device,
                                    "gpu_name": gpu_name,
                                    "batch_size": bs,
                                    "num_threads": nt if nt > 0 else "",
                                    "throughput_mean": f"{stats['throughput_mean']:.2f}",
                                    "throughput_std": f"{stats['throughput_std']:.2f}",
                                    "throughput_median": f"{stats['throughput_median']:.2f}",
                                    "throughput_min": f"{stats['throughput_min']:.2f}",
                                    "throughput_max": f"{stats['throughput_max']:.2f}",
                                    "pixels_per_sec": f"{pixels_per_sec:.0f}",
                                    "latency_mean_ms": f"{stats['latency_mean_ms']:.3f}",
                                    "latency_std_ms": f"{stats['latency_std_ms']:.3f}",
                                    "params_M": f"{task_params:.2f}",
                                    "macs_G": f"{task_macs:.2f}",
                                    "peak_memory_mb": f"{stats['peak_memory_mb']:.1f}",
                                    "pytorch_version": torch.__version__,
                                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                }
                                writer.writerow(row)
                                csv_file.flush()

                                n = stats["num_iterations"]
                                tp = stats["throughput_mean"]
                                print(f"{tp:.1f} img/s "
                                      f"(±{stats['throughput_std']:.1f}, "
                                      f"n={n})")

                            except torch.cuda.OutOfMemoryError:
                                print("OOM")
                                row = {
                                    "model_name": mc.timm_name,
                                    "display_name": mc.display_name,
                                    "model_family": mc.family,
                                    "model_type": mc.arch_type,
                                    "task": task,
                                    "precision": prec,
                                    "compiled": cm != "none",
                                    "compile_mode": cm,
                                    "device": args.device,
                                    "gpu_name": gpu_name,
                                    "batch_size": bs,
                                    "num_threads": nt if nt > 0 else "",
                                    "throughput_mean": "OOM",
                                    "throughput_std": "",
                                    "throughput_median": "",
                                    "throughput_min": "",
                                    "throughput_max": "",
                                    "pixels_per_sec": "",
                                    "latency_mean_ms": "",
                                    "latency_std_ms": "",
                                    "params_M": f"{task_params:.2f}",
                                    "macs_G": f"{task_macs:.2f}",
                                    "peak_memory_mb": "",
                                    "pytorch_version": torch.__version__,
                                    "timestamp": time.strftime(
                                        "%Y-%m-%dT%H:%M:%S"),
                                }
                                writer.writerow(row)
                                csv_file.flush()
                                gpu_cleanup()
                            except Exception as e:
                                print(f"ERROR: {e}")
                            finally:
                                # Clean up model
                                try:
                                    del model
                                except NameError:
                                    pass
                                try:
                                    del dl
                                except (NameError, UnboundLocalError):
                                    pass
                                gpu_cleanup()

                        # Small sleep between compile/precision combos
                        time.sleep(1)

        # Longer sleep between models
        time.sleep(2)

    csv_file.close()
    print(f"\n✅ Results saved to {output_path}")
    print(f"   {completed} configurations benchmarked")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="GeoSpeedy: Geospatial model throughput benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
                    help="Device to benchmark on (default: cuda)")
    p.add_argument("--gpu-id", type=int, default=0,
                    help="GPU index to use (default: 0)")
    p.add_argument("--models", nargs="+", default=None,
                    help="Filter to specific timm model names")
    p.add_argument("--tasks", nargs="+",
                    default=["classification", "segmentation"],
                    choices=["classification", "segmentation"],
                    help="Tasks to benchmark")
    p.add_argument("--precisions", nargs="+", default=["fp32", "fp16", "amp"],
                    choices=["fp32", "fp16", "amp"],
                    help="Precision modes")
    p.add_argument("--compile-modes", nargs="+", default=["none", "default"],
                    choices=["none", "default", "max-autotune"],
                    help="torch.compile modes")
    p.add_argument("--batch-sizes", nargs="+", type=int,
                    default=[1, 8, 32, 64],
                    help="Batch sizes to test")
    p.add_argument("--num-threads", nargs="+", type=int, default=None,
                    help="CPU thread counts (CPU mode only)")
    p.add_argument("--num-workers", type=int, default=4,
                    help="DataLoader workers (default: 4)")
    p.add_argument("--warmup", type=int, default=20,
                    help="Number of warmup iterations (default: 20)")
    p.add_argument("--timed-seconds", type=float, default=10.0,
                    help="Minimum seconds to time (default: 10)")
    p.add_argument("--output", type=str,
                    default="results/benchmark_results.csv",
                    help="Output CSV path")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args)
