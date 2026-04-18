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

    # Include DataLoader overhead (default is pure GPU compute)
    python benchmark.py --gpu-id 0 --dataloader
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
    "pixels_per_sec",
    "latency_mean_ms",
    "latency_p50_ms",
    "latency_p95_ms",
    "latency_p99_ms",
    "params_M",
    "macs_G",
    "peak_memory_mb",
    "tf32_enabled",
    "input_channels",
    "input_size",
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
        "cudnn_version": (
            torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        ),
    }

    # Extended GPU metadata via nvidia-smi
    smi_queries = {
        "driver_version": "driver_version",
        "persistence_mode": "persistence_mode",
        "power_limit_w": "power.limit",
        "clock_max_sm_mhz": "clocks.max.sm",
        "clock_max_mem_mhz": "clocks.max.mem",
    }
    for key, query in smi_queries.items():
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--query-gpu={query}",
                    "--format=csv,noheader,nounits",
                    f"--id={gpu_id}",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            info[key] = result.stdout.strip()
        except Exception:
            info[key] = None

    # Git SHA
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        info["git_sha"] = result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        info["git_sha"] = None

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
    torch._dynamo.reset()


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


def create_model_for_task(
    config: ModelConfig,
    task: str,
    device: torch.device,
    input_channels: int = 3,
    input_size: int = 224,
) -> nn.Module | None:
    """Instantiate a model for the given task. Returns None if unsupported."""
    if config.source == "geo":
        if task == "segmentation":
            return None  # Geo models are encoder-only
        from geo_models import create_geo_model

        return create_geo_model(config.geo_model_key, device)

    if task == "classification":
        model = timm.create_model(
            config.timm_name, pretrained=False, num_classes=10, in_chans=input_channels
        )
    elif task == "segmentation":
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = smp.DPT(
                encoder_name=config.smp_encoder_name,
                encoder_weights=None,
                in_channels=input_channels,
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
    elif precision == "bf16":
        model = model.bfloat16()
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
    precision: str = "fp32",
    max_power: int = 20,
    num_validate: int = 3,
    input_channels: int = 3,
    input_size: int = 224,
    start_power: int = 0,
) -> int:
    """Find largest power-of-2 batch size that fits in GPU memory.

    Creates a fresh model at the given precision, tries increasing powers of 2,
    runs ``num_validate`` forward passes at each size to account for
    cudnn autotuner memory, and cleans up after each OOM.
    Returns the largest successful batch size.
    """
    max_bs = 1

    use_amp = precision == "amp"

    for power in range(start_power, max_power + 1):
        bs = 2**power
        gpu_cleanup()
        try:
            model = create_model_for_task(
                config, task, device,
                input_channels=input_channels, input_size=input_size,
            )
            if model is None:
                return 0
            model = apply_precision(model, precision)
            x = torch.randn(bs, input_channels, input_size, input_size, device=device)
            if precision == "fp16":
                x = x.half()
            elif precision == "bf16":
                x = x.bfloat16()
            with torch.no_grad():
                for _ in range(num_validate):
                    if use_amp:
                        with torch.amp.autocast("cuda"):
                            _ = model(x)
                    else:
                        _ = model(x)
            torch.cuda.synchronize()
            max_bs = bs
            del model, x
            gpu_cleanup()
        except torch.cuda.OutOfMemoryError:
            break
        except RuntimeError as e:
            msg = str(e)
            if (
                "canUse32BitIndexMath" in msg
                or "32-bit indexing" in msg
                or "INT_MAX" in msg
            ):
                break
            raise

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

    # --- warmup ---
    data_iter = iter(dataloader)
    for _ in range(num_warmup):
        try:
            images, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            images, _ = next(data_iter)
        images = images.to(device, non_blocking=True)
        if precision == "fp16":
            images = images.half()
        elif precision == "bf16":
            images = images.bfloat16()
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
    batch_times: list[float] = []
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
        if precision == "fp16":
            images = images.half()
        elif precision == "bf16":
            images = images.bfloat16()

        t_batch = time.perf_counter()
        with torch.no_grad():
            if use_amp:
                with torch.amp.autocast("cuda"):
                    _ = model(images)
            else:
                _ = model(images)
        torch.cuda.synchronize()
        batch_times.append(time.perf_counter() - t_batch)

        total_images += batch_size

        if time.perf_counter() - t_start >= min_timed_seconds:
            break

    elapsed_s = time.perf_counter() - t_start

    return _format_gpu_stats(total_images, batch_size, elapsed_s, batch_times)


def benchmark_gpu_preallocated(
    model: nn.Module,
    batch_size: int,
    precision: str,
    device: torch.device,
    num_warmup: int = 20,
    min_timed_seconds: float = 30.0,
    input_channels: int = 3,
    input_size: int = 224,
) -> dict:
    """Benchmark on GPU with a pre-allocated batch (no DataLoader overhead)."""
    use_amp = precision == "amp"

    images = torch.randn(batch_size, input_channels, input_size, input_size, device=device)
    if precision == "fp16":
        images = images.half()
    elif precision == "bf16":
        images = images.bfloat16()

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
    batch_times: list[float] = []
    torch.cuda.synchronize()
    t_start = time.perf_counter()

    while True:
        t_batch = time.perf_counter()
        with torch.no_grad():
            if use_amp:
                with torch.amp.autocast("cuda"):
                    _ = model(images)
            else:
                _ = model(images)
        torch.cuda.synchronize()
        batch_times.append(time.perf_counter() - t_batch)
        total_images += batch_size
        if time.perf_counter() - t_start >= min_timed_seconds:
            break

    elapsed_s = time.perf_counter() - t_start
    del images

    return _format_gpu_stats(total_images, batch_size, elapsed_s, batch_times)


def _format_gpu_stats(
    total_images: int, batch_size: int, elapsed_s: float, batch_times: list[float]
) -> dict:
    throughput = total_images / elapsed_s
    peak_mem = torch.cuda.max_memory_allocated() / 1e6
    latency_mean_ms = elapsed_s / (total_images / batch_size) * 1000

    if batch_times:
        import numpy as np

        batch_ms = np.array(batch_times) * 1000
        p50 = float(np.percentile(batch_ms, 50))
        p95 = float(np.percentile(batch_ms, 95))
        p99 = float(np.percentile(batch_ms, 99))
    else:
        p50 = p95 = p99 = latency_mean_ms

    return {
        "throughput_mean": float(throughput),
        "latency_mean_ms": float(latency_mean_ms),
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
        "latency_p99_ms": p99,
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
    tf32_enabled: bool = False,
    input_channels: int = 3,
    input_size: int = 224,
) -> dict | None:
    """Run a single benchmark config. Returns a CSV row dict or None."""
    gpu_cleanup()
    model = None
    dl = None
    try:
        model = create_model_for_task(
            mc, task, device,
            input_channels=input_channels, input_size=input_size,
        )
        if model is None:
            return None
        model = apply_precision(model, precision)
        model, compile_ok = apply_compile(model, compile_mode)
        actual_compile_mode = compile_mode if compile_ok else "none"
        actual_compiled = compile_ok and compile_mode != "none"

        # Compute seg-specific params and MACs (U-Net wraps the backbone)
        task_macs, task_params = macs_g, params_m
        if task == "segmentation":
            task_params = count_params(model)
            try:
                # Estimate MACs on a fresh fp32 CPU copy to avoid device/dtype issues
                seg_model_cpu = create_model_for_task(
                    mc, task, torch.device("cpu"),
                    input_channels=input_channels,
                    input_size=input_size,
                )
                if seg_model_cpu is not None:
                    seg_shape = (1, input_channels, input_size, input_size)
                    task_macs = estimate_macs(
                        seg_model_cpu, input_shape=seg_shape, device="cpu"
                    )
                    del seg_model_cpu
            except Exception:
                pass  # fall back to backbone MACs

        torch.cuda.reset_peak_memory_stats()
        if not args.dataloader:
            stats = benchmark_gpu_preallocated(
                model,
                batch_size,
                precision,
                device,
                num_warmup=args.warmup,
                min_timed_seconds=args.timed_seconds,
                input_channels=input_channels,
                input_size=input_size,
            )
        else:
            dl = create_dataloader(
                task=task,
                batch_size=batch_size,
                num_workers=8,
                prefetch_factor=2,
                length=max(batch_size * 500, 10_000),
                channels=input_channels,
                size=input_size,
            )
            stats = benchmark_gpu(
                model,
                dl,
                precision,
                device,
                num_warmup=args.warmup,
                min_timed_seconds=args.timed_seconds,
            )

        pixels_per_sec = stats["throughput_mean"] * input_size * input_size
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
            "pixels_per_sec": f"{pixels_per_sec:.0f}",
            "latency_mean_ms": f"{stats['latency_mean_ms']:.3f}",
            "latency_p50_ms": f"{stats['latency_p50_ms']:.3f}",
            "latency_p95_ms": f"{stats['latency_p95_ms']:.3f}",
            "latency_p99_ms": f"{stats['latency_p99_ms']:.3f}",
            "params_M": f"{task_params:.2f}",
            "macs_G": f"{task_macs:.2f}",
            "peak_memory_mb": f"{stats['peak_memory_mb']:.1f}",
            "tf32_enabled": tf32_enabled,
            "input_channels": input_channels,
            "input_size": input_size,
            "pytorch_version": torch.__version__,
            "cuda_version": get_cuda_version(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    except torch.cuda.OutOfMemoryError:
        return "OOM"
    except RuntimeError as e:
        # torch.compile is lazy — Triton/inductor errors surface on first
        # forward pass, not at compile() time.  Catch them so the rest of
        # the benchmark can continue.
        from torch._dynamo.exc import BackendCompilerFailed

        if isinstance(e, BackendCompilerFailed):
            print(f"\n    ⚠ torch.compile failed at runtime (skipping): {e}")
            return "COMPILE_ERROR"
        msg = str(e)
        if "canUse32BitIndexMath" in msg or "32-bit indexing" in msg or "INT_MAX" in msg:
            print(
                "\n    ⚠ Batch size exceeds cuDNN 32-bit indexing limit "
                "(~2.147B elements); skipping."
            )
            return "OOM"
        raise
    finally:
        # Explicitly shut down DataLoader workers to free file descriptors
        if dl is not None and hasattr(dl, "_iterator") and dl._iterator is not None:
            dl._iterator._shutdown_workers()
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

    # Enable TF32 for fp32 matmuls (Ampere+). Matches what virtually all
    # "fp32" benchmarks on modern NVIDIA hardware actually measure; without
    # this, the fp32 precision mode runs strict IEEE-754 and looks
    # artificially slow on H100/A100.
    torch.set_float32_matmul_precision("high")
    tf32_enabled = torch.cuda.get_device_capability(0) >= (8, 0)
    tf32_label = "enabled" if tf32_enabled else "not available"
    print(f'🔢 float32 matmul precision: "high" (TF32 {tf32_label} on this GPU)')

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
        output_path = Path(f"results/v3/{slug}.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save hardware info
    hw_info = collect_hardware_info(args.gpu_id)
    hw_path = output_path.parent / (output_path.stem + "_hardware.json")
    with open(hw_path, "w") as f:
        json.dump(hw_info, f, indent=2)
    print(f"💾 Hardware info: {hw_path}")

    model_configs = get_models(args.models if args.models else None)
    input_channels = args.input_channels
    input_size = args.input_size
    print(f"📋 Models: {len(model_configs)}")
    print(f"📋 Tasks: {args.tasks}")
    print(f"📋 Precisions: {args.precisions}")
    print(f"📋 Compile modes: {args.compile_modes}")
    print(f"📋 Input: {input_channels}×{input_size}×{input_size}")
    if auto_batch:
        print("📋 Batch size: auto (largest power-of-2 that fits, per precision)")
    else:
        print(f"📋 Batch sizes: {args.batch_sizes}")
    print(f"📋 Timed seconds: {args.timed_seconds}")

    # Load existing results to skip already-completed configs
    file_exists = output_path.exists() and output_path.stat().st_size > 0
    completed_keys: set[tuple] = set()
    if file_exists:
        try:
            import pandas as pd

            existing = pd.read_csv(output_path)
            for _, row in existing.iterrows():
                key = (
                    row.get("model_name"),
                    row.get("task"),
                    row.get("precision"),
                    row.get("compile_mode"),
                )
                completed_keys.add(key)
            print(f"📂 Found {len(completed_keys)} existing configs, will skip")
        except Exception:
            pass

    # Open CSV writer
    csv_file = open(output_path, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
    if not file_exists:
        writer.writeheader()

    completed = 0

    for mc in model_configs:
        print(f"\n{'=' * 70}")
        print(f"  {mc.display_name} ({mc.timm_name}) — {mc.arch_type}")
        print(f"{'=' * 70}")

        # For geo models, use their native input dimensions
        if mc.source == "geo":
            model_channels = mc.native_channels
            model_size = mc.native_size
        else:
            model_channels = input_channels
            model_size = input_size

        # Compute MACs once on CPU
        macs_g, params_m = -1.0, -1.0
        try:
            if mc.source == "geo":
                from geo_models import create_geo_model

                tmp = create_geo_model(mc.geo_model_key, torch.device("cpu"))
                params_m = count_params(tmp)
                cls_shape = (1, model_channels, model_size, model_size)
                macs_g = estimate_macs(tmp, input_shape=cls_shape, device="cpu")
            else:
                tmp = timm.create_model(
                    mc.timm_name, pretrained=False, num_classes=10,
                    in_chans=model_channels,
                )
                tmp.eval()
                params_m = count_params(tmp)
                cls_shape = (1, model_channels, model_size, model_size)
                macs_g = estimate_macs(tmp, input_shape=cls_shape, device="cpu")
            del tmp
            gc.collect()
        except Exception as e:
            print(f"  ⚠ Could not compute MACs: {e}")

        for task in args.tasks:
            # Skip segmentation for geo models (encoder-only)
            if task == "segmentation" and mc.source == "geo":
                print("  ⏭ Skipping segmentation (geo model, encoder-only)")
                continue

            # Track last successful BS power for smarter probing across precisions
            last_max_power = 0

            for prec in args.precisions:
                # Skip bf16 on GPUs that don't support it (pre-Ampere)
                if prec == "bf16" and not tf32_enabled:
                    print(f"  ⏭ Skipping {prec} (not supported on this GPU)")
                    continue

                # Skip non-fp32 for geo models that only support fp32
                if prec != "fp32" and mc.source == "geo":
                    from geo_models import GEO_MODEL_REGISTRY

                    entry = GEO_MODEL_REGISTRY.get(mc.geo_model_key, {})
                    if entry.get("fp32_only", False):
                        print(f"  ⏭ Skipping {prec} (model only supports fp32)")
                        continue

                # Determine batch sizes for this model+task+precision
                if auto_batch:
                    print(f"  🔍 Finding max batch size for {task}/{prec}...", end=" ", flush=True)
                    max_bs = find_max_batch_size(
                        mc, task, device,
                        precision=prec,
                        input_channels=model_channels,
                        input_size=model_size,
                        start_power=max(0, last_max_power - 1),
                    )
                    if max_bs == 0:
                        print("SKIP (model unsupported)")
                        continue
                    print(f"bs={max_bs}")
                    batch_sizes_to_run = [max_bs]
                    # Remember the power for the next precision probe
                    last_max_power = max_bs.bit_length() - 1
                elif args.batch_sizes:
                    batch_sizes_to_run = args.batch_sizes
                else:
                    batch_sizes_to_run = [32]

                for cm in args.compile_modes:
                    for bs in batch_sizes_to_run:
                        completed += 1
                        label = f"  [{completed}] {task} | {prec} | compile={cm} | bs={bs}"

                        config_key = (mc.timm_name, task, prec, cm)
                        if config_key in completed_keys:
                            print(label, "... SKIP (already in CSV)")
                            continue

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
                            tf32_enabled=tf32_enabled,
                            input_channels=model_channels,
                            input_size=model_size,
                        )
                        if result == "COMPILE_ERROR":
                            print("SKIP (compile error)")
                            continue

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
                                    tf32_enabled=tf32_enabled,
                                    input_channels=model_channels,
                                    input_size=model_size,
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
        default=["fp32", "fp16", "amp", "bf16"],
        choices=["fp32", "fp16", "amp", "bf16"],
        help="Precision modes (bf16 auto-skipped on pre-Ampere GPUs)",
    )
    p.add_argument(
        "--compile-modes",
        nargs="+",
        default=["none", "default"],
        choices=["none", "default", "max-autotune"],
        help="torch.compile modes to benchmark (default: none + default)",
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
        "--dataloader",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use a PyTorch DataLoader to feed data (adds realistic pipeline "
        "overhead). Default is a pre-allocated GPU batch, which measures "
        "peak compute throughput.",
    )
    p.add_argument(
        "--input-channels",
        type=int,
        default=3,
        help="Number of input channels (default: 3). Use 4/6/13 for multispectral EO data.",
    )
    p.add_argument(
        "--input-size",
        type=int,
        default=224,
        help="Spatial input size (default: 224). Images are input_size × input_size.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args)
