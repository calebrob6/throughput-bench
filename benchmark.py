#!/usr/bin/env python3
"""ThroughputBencher — rigorous throughput benchmarking for geospatial model backbones.

Measures inference throughput (images/sec) for classification and segmentation
across many model architectures, precision modes, and hardware configurations.

Usage examples:
    # Full benchmark on GPU 0 (auto batch size, 30s per config)
    python benchmark.py --gpu-id 0

    # Quick test with one model
    python benchmark.py --gpu-id 0 --models resnet50

    # Manual batch size sweep
    python benchmark.py --gpu-id 0 --batch-sizes 1 8 32 64

    # Only classification, AMP precision
    python benchmark.py --gpu-id 0 --tasks classification --precisions amp

    # Pure GPU compute (no DataLoader overhead)
    python benchmark.py --gpu-id 0 --no-dataloader
"""

import argparse
import csv
import gc
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import segmentation_models_pytorch as smp
import timm
import torch
import torch.nn as nn

from data import create_dataloader
from models import ModelConfig, get_models

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
    "gpu_name",
    "gpu_mem_gb",
    "batch_size",
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
    "cuda_version",
    "timestamp",
]

# ---------------------------------------------------------------------------
# Hardware helpers
# ---------------------------------------------------------------------------


def get_gpu_name(gpu_id: int = 0) -> str:
    return torch.cuda.get_device_name(gpu_id)


def get_gpu_mem_gb(gpu_id: int = 0) -> float:
    return torch.cuda.get_device_properties(gpu_id).total_memory / 1e9


def get_gpu_slug() -> str:
    """Sanitized GPU name for filenames, e.g. 'tesla_v100_sxm2_32gb'."""
    name = get_gpu_name(0)
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()
    return slug


def get_cuda_version() -> str:
    return torch.version.cuda or "N/A"


def collect_hardware_info(gpu_id: int) -> dict:
    """Collect full hardware metadata."""
    info = {
        "gpu_name": get_gpu_name(0),
        "gpu_mem_gb": round(get_gpu_mem_gb(0), 1),
        "gpu_id_physical": gpu_id,
        "cuda_version": get_cuda_version(),
        "pytorch_version": torch.__version__,
        "timm_version": timm.__version__,
        "smp_version": smp.__version__,
        "python_version": platform.python_version(),
        "os": platform.system(),
        "cpu": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
    }
    return info


def check_gpu_free(gpu_id: int) -> bool:
    """Check that the target GPU has no other processes running."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        gpu_uuids_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        target_uuid = None
        for line in gpu_uuids_result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2 and parts[0] == str(gpu_id):
                target_uuid = parts[1]
                break
        if target_uuid is None:
            return True
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
        return True


def gpu_cleanup():
    """Free GPU memory between benchmark runs."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------


def count_params(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def estimate_macs(
    model: nn.Module, input_shape: tuple = (1, 3, 224, 224), device: str = "cpu"
) -> float:
    from torch.utils.flop_counter import FlopCounterMode

    inp = torch.randn(*input_shape, device=device)
    with FlopCounterMode(display=False) as fcm:
        model(inp)
    return fcm.get_total_flops() / 1e9


def create_model_for_task(config: ModelConfig, task: str, device: torch.device) -> nn.Module | None:
    """Instantiate a model for the given task. Returns None if unsupported."""
    if task == "classification":
        model = timm.create_model(config.timm_name, pretrained=False, num_classes=10)
    elif task == "segmentation":
        if not config.supports_segmentation:
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


def apply_precision(model: nn.Module, precision: str) -> nn.Module:
    if precision == "fp16":
        model = model.half()
    return model


def apply_compile(model: nn.Module, compile_mode: str) -> tuple[nn.Module, bool]:
    if compile_mode == "none":
        return model, True
    try:
        model = torch.compile(model, mode=compile_mode)
        return model, True
    except Exception as e:
        print(f"    ⚠ torch.compile({compile_mode}) failed: {e}")
        return model, False


# ---------------------------------------------------------------------------
# Auto batch size detection
# ---------------------------------------------------------------------------


def find_max_batch_size(
    config: ModelConfig,
    task: str,
    device: torch.device,
    max_power: int = 9,
    num_validate: int = 3,
) -> int:
    """Find largest power-of-2 batch size that fits in GPU memory.

    Creates a fresh fp32 uncompiled model, tries increasing powers of 2,
    runs ``num_validate`` forward passes at each size to account for
    cudnn autotuner memory, and cleans up after each OOM.
    Returns the largest successful batch size.
    """
    max_bs = 1

    for power in range(max_power + 1):  # 1, 2, 4, ..., 512
        bs = 2**power
        gpu_cleanup()
        try:
            model = create_model_for_task(config, task, device)
            if model is None:
                return 0
            x = torch.randn(bs, 3, 224, 224, device=device)
            with torch.no_grad():
                for _ in range(num_validate):
                    _ = model(x)
            torch.cuda.synchronize()
            max_bs = bs
            del model, x
            gpu_cleanup()
        except torch.cuda.OutOfMemoryError:
            break

    return max_bs


# ---------------------------------------------------------------------------
# Benchmarking core
# ---------------------------------------------------------------------------


def benchmark_gpu(
    model: nn.Module,
    dataloader,
    precision: str,
    device: torch.device,
    num_warmup: int = 20,
    min_timed_seconds: float = 30.0,
) -> dict:
    """Benchmark on GPU using wall-clock timing with DataLoader."""
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
    total_images = 0
    data_iter = iter(dataloader)

    torch.cuda.synchronize()
    t_start = time.perf_counter()

    while True:
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

        total_images += batch_size

        if time.perf_counter() - t_start >= min_timed_seconds:
            break

    torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - t_start

    return _format_gpu_stats(total_images, batch_size, elapsed_s)


def benchmark_gpu_preallocated(
    model: nn.Module,
    batch_size: int,
    precision: str,
    device: torch.device,
    num_warmup: int = 20,
    min_timed_seconds: float = 30.0,
) -> dict:
    """Benchmark on GPU with a pre-allocated batch (no DataLoader overhead)."""
    use_amp = precision == "amp"
    use_fp16_input = precision == "fp16"

    images = torch.randn(batch_size, 3, 224, 224, device=device)
    if use_fp16_input:
        images = images.half()

    # --- warmup ---
    for _ in range(num_warmup):
        with torch.no_grad():
            if use_amp:
                with torch.amp.autocast("cuda"):
                    _ = model(images)
            else:
                _ = model(images)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()

    # --- timed ---
    total_images = 0
    torch.cuda.synchronize()
    t_start = time.perf_counter()

    while True:
        with torch.no_grad():
            if use_amp:
                with torch.amp.autocast("cuda"):
                    _ = model(images)
            else:
                _ = model(images)
        total_images += batch_size
        if time.perf_counter() - t_start >= min_timed_seconds:
            break

    torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - t_start
    del images

    return _format_gpu_stats(total_images, batch_size, elapsed_s)


def _format_gpu_stats(total_images: int, batch_size: int, elapsed_s: float) -> dict:
    throughput = total_images / elapsed_s
    peak_mem = torch.cuda.max_memory_allocated() / 1e6
    return {
        "throughput_mean": float(throughput),
        "throughput_std": 0.0,
        "throughput_median": float(throughput),
        "throughput_min": float(throughput),
        "throughput_max": float(throughput),
        "latency_mean_ms": float(elapsed_s / (total_images / batch_size) * 1000),
        "latency_std_ms": 0.0,
        "peak_memory_mb": float(peak_mem),
        "num_iterations": total_images // batch_size,
    }


# ---------------------------------------------------------------------------
# Single benchmark run helper
# ---------------------------------------------------------------------------


def run_single_benchmark(
    mc: ModelConfig,
    task: str,
    precision: str,
    compile_mode: str,
    batch_size: int,
    device: torch.device,
    args,
    gpu_name: str,
    gpu_mem_gb: float,
    macs_g: float,
    params_m: float,
) -> dict | None:
    """Run a single benchmark config. Returns a CSV row dict or None."""
    gpu_cleanup()
    model = None
    dl = None
    try:
        model = create_model_for_task(mc, task, device)
        if model is None:
            return None
        model = apply_precision(model, precision)
        model, compile_ok = apply_compile(model, compile_mode)
        actual_compile_mode = compile_mode if compile_ok else "none"
        actual_compiled = compile_ok and compile_mode != "none"

        # Compute seg-specific params
        task_macs, task_params = macs_g, params_m
        if task == "segmentation":
            task_params = count_params(model)

        torch.cuda.reset_peak_memory_stats()
        if args.no_dataloader:
            stats = benchmark_gpu_preallocated(
                model,
                batch_size,
                precision,
                device,
                num_warmup=args.warmup,
                min_timed_seconds=args.timed_seconds,
            )
        else:
            dl = create_dataloader(
                task=task,
                batch_size=batch_size,
                num_workers=8,
                prefetch_factor=2,
                length=max(batch_size * 500, 10_000),
            )
            stats = benchmark_gpu(
                model,
                dl,
                precision,
                device,
                num_warmup=args.warmup,
                min_timed_seconds=args.timed_seconds,
            )

        pixels_per_sec = stats["throughput_mean"] * 224 * 224
        return {
            "model_name": mc.timm_name,
            "display_name": mc.display_name,
            "model_family": mc.family,
            "model_type": mc.arch_type,
            "task": task,
            "precision": precision,
            "compiled": actual_compiled,
            "compile_mode": actual_compile_mode,
            "gpu_name": gpu_name,
            "gpu_mem_gb": f"{gpu_mem_gb:.1f}",
            "batch_size": batch_size,
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
            "cuda_version": get_cuda_version(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    except torch.cuda.OutOfMemoryError:
        return "OOM"
    finally:
        del model, dl
        gpu_cleanup()


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------


def run_benchmark(args):
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    auto_batch = args.batch_sizes is None

    gpu_id = args.gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0")
    gpu_name = get_gpu_name(0)
    gpu_mem_gb = get_gpu_mem_gb(0)
    print(f"🖥  Device: GPU {gpu_id} ({gpu_name}, {gpu_mem_gb:.0f} GB)")

    if not check_gpu_free(gpu_id):
        if args.force:
            print(
                f"⚠  WARNING: Other processes detected on GPU {gpu_id}. "
                f"Results may be unreliable. (--force used, continuing)"
            )
        else:
            print(f"❌ ERROR: Other processes detected on GPU {gpu_id}.")
            print("   Benchmarks require an idle GPU for reliable results.")
            print("   Use --force to override this check.")
            sys.exit(1)

    # Auto-detect output path if user didn't specify
    if args.output == "auto":
        slug = get_gpu_slug()
        output_path = Path(f"results/{slug}.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save hardware info
    hw_info = collect_hardware_info(args.gpu_id)
    hw_path = output_path.parent / (output_path.stem + "_hardware.json")
    with open(hw_path, "w") as f:
        json.dump(hw_info, f, indent=2)
    print(f"💾 Hardware info: {hw_path}")

    model_configs = get_models(args.models if args.models else None)
    print(f"📋 Models: {len(model_configs)}")
    print(f"📋 Tasks: {args.tasks}")
    print(f"📋 Precisions: {args.precisions}")
    print(f"📋 Compile modes: {args.compile_modes}")
    if auto_batch:
        print("📋 Batch size: auto (largest power-of-2 that fits)")
    else:
        print(f"📋 Batch sizes: {args.batch_sizes}")
    print(f"📋 Timed seconds: {args.timed_seconds}")

    # Open CSV writer
    file_exists = output_path.exists() and output_path.stat().st_size > 0
    csv_file = open(output_path, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
    if not file_exists:
        writer.writeheader()

    completed = 0

    for mc in model_configs:
        print(f"\n{'=' * 70}")
        print(f"  {mc.display_name} ({mc.timm_name}) — {mc.arch_type}")
        print(f"{'=' * 70}")

        # Compute MACs once on CPU
        macs_g, params_m = -1.0, -1.0
        try:
            tmp = timm.create_model(mc.timm_name, pretrained=False, num_classes=10)
            tmp.eval()
            params_m = count_params(tmp)
            macs_g = estimate_macs(tmp, device="cpu")
            del tmp
            gc.collect()
        except Exception as e:
            print(f"  ⚠ Could not compute MACs: {e}")

        for task in args.tasks:
            if task == "segmentation" and not mc.supports_segmentation:
                print("  ⏭ Skipping segmentation (not supported)")
                continue

            # Determine batch sizes for this model+task
            if auto_batch:
                print(f"  🔍 Finding max batch size for {task}...", end=" ", flush=True)
                max_bs = find_max_batch_size(mc, task, device)
                if max_bs == 0:
                    print("SKIP (model unsupported)")
                    continue
                print(f"bs={max_bs}")
                batch_sizes_to_run = [max_bs]
            elif args.batch_sizes:
                batch_sizes_to_run = args.batch_sizes
            else:
                batch_sizes_to_run = [32]

            for prec in args.precisions:
                for cm in args.compile_modes:
                    for bs in batch_sizes_to_run:
                        completed += 1
                        label = f"  [{completed}] {task} | {prec} | compile={cm} | bs={bs}"
                        print(label, end=" ... ", flush=True)

                        result = run_single_benchmark(
                            mc,
                            task,
                            prec,
                            cm,
                            bs,
                            device,
                            args,
                            gpu_name,
                            gpu_mem_gb,
                            macs_g,
                            params_m,
                        )
                        if result == "OOM":
                            # Step down batch size for compiled mode
                            if cm != "none" and bs > 1:
                                stepped = bs // 2
                                print(
                                    f"OOM → retrying bs={stepped}",
                                    end=" ... ",
                                    flush=True,
                                )
                                result = run_single_benchmark(
                                    mc,
                                    task,
                                    prec,
                                    cm,
                                    stepped,
                                    device,
                                    args,
                                    gpu_name,
                                    gpu_mem_gb,
                                    macs_g,
                                    params_m,
                                )
                            if result == "OOM" or result is None:
                                print("OOM")
                                # Write OOM row
                                writer.writerow(
                                    {
                                        "model_name": mc.timm_name,
                                        "display_name": mc.display_name,
                                        "model_family": mc.family,
                                        "model_type": mc.arch_type,
                                        "task": task,
                                        "precision": prec,
                                        "compiled": cm != "none",
                                        "compile_mode": cm,
                                        "gpu_name": gpu_name,
                                        "gpu_mem_gb": f"{gpu_mem_gb:.1f}",
                                        "batch_size": bs,
                                        "throughput_mean": "OOM",
                                        **{
                                            c: ""
                                            for c in CSV_COLUMNS
                                            if c
                                            not in {
                                                "model_name",
                                                "display_name",
                                                "model_family",
                                                "model_type",
                                                "task",
                                                "precision",
                                                "compiled",
                                                "compile_mode",
                                                "gpu_name",
                                                "gpu_mem_gb",
                                                "batch_size",
                                                "throughput_mean",
                                            }
                                        },
                                    }
                                )
                                csv_file.flush()
                                continue

                        if result and isinstance(result, dict):
                            tp = float(result["throughput_mean"])
                            print(f"{tp:.1f} img/s")
                            writer.writerow(result)
                            csv_file.flush()
                        elif result is None:
                            print("SKIP")

                    time.sleep(1)

        time.sleep(2)

    csv_file.close()
    print(f"\n✅ Results saved to {output_path}")
    print(f"   {completed} configurations benchmarked")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="ThroughputBencher: Geospatial model throughput benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--gpu-id", type=int, default=0, help="GPU index to use (default: 0)")
    p.add_argument("--models", nargs="+", default=None, help="Filter to specific timm model names")
    p.add_argument(
        "--tasks",
        nargs="+",
        default=["classification", "segmentation"],
        choices=["classification", "segmentation"],
        help="Tasks to benchmark",
    )
    p.add_argument(
        "--precisions",
        nargs="+",
        default=["fp32", "fp16", "amp"],
        choices=["fp32", "fp16", "amp"],
        help="Precision modes",
    )
    p.add_argument(
        "--compile-modes",
        nargs="+",
        default=["none", "default"],
        choices=["none", "default", "max-autotune"],
        help="torch.compile modes",
    )
    p.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=None,
        help="Manual batch sizes (default: auto-detect max power-of-2 that fits in GPU memory)",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Number of warmup iterations (default: 20)",
    )
    p.add_argument(
        "--timed-seconds",
        type=float,
        default=30.0,
        help="Minimum seconds to time (default: 30)",
    )
    p.add_argument(
        "--output",
        type=str,
        default="auto",
        help="Output CSV path (default: auto-detect from GPU)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Run even if other processes are using the GPU",
    )
    p.add_argument(
        "--no-dataloader",
        action="store_true",
        help="Use pre-allocated GPU batch instead of DataLoader "
        "(maximizes GPU utilization, measures pure compute)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args)
