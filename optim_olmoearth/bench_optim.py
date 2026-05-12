"""Benchmark all OlmoEarth-Base inference variants on a single GPU.

Each variant is validated for equivalence against the reference once,
then timed using the same protocol as ``benchmark.py`` (pre-allocated GPU
batch, 20 warmup iters + ≥30 s timed, peak memory tracked).

Variants currently exercised:
    * reference                         — geo_models.OlmoEarthWrapper
    * reference + torch.compile(default)
    * reference + torch.compile(max-autotune)
    * fast                              — FastOlmoEarthBase
    * fast + torch.compile(default)
    * fast + torch.compile(max-autotune)
    * fast + CUDA Graphs                — captured fixed-shape graph
    * fast + ONNX Runtime (if installed)

Outputs go to ``optim_olmoearth/results/<gpu_slug>.csv`` so they don't
clobber the canonical ``results/<gpu_slug>.csv``.
"""

from __future__ import annotations

import argparse
import csv
import gc
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optim_olmoearth.equiv import check_equivalence  # noqa: E402
from optim_olmoearth.fast_wrapper import FastOlmoEarthBase  # noqa: E402
from optim_olmoearth.reference_set import (  # noqa: E402
    REF_SEED,
    _seeded_input,
    _seeded_wrapper,
)

# ---------------------------------------------------------------------------
# Timing helpers (mirrors benchmark.py)
# ---------------------------------------------------------------------------


def _time_callable(
    fwd,
    batch_size: int,
    warmup: int = 20,
    timed_seconds: float = 30.0,
) -> dict:
    """Return throughput, mean/p50/p95/p99 latency, peak memory."""
    for _ in range(warmup):
        fwd()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    times: list[float] = []
    total_images = 0
    t_start = time.perf_counter()
    while True:
        t_batch = time.perf_counter()
        fwd()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t_batch)
        total_images += batch_size
        if time.perf_counter() - t_start >= timed_seconds:
            break
    elapsed = time.perf_counter() - t_start

    import numpy as np

    times_ms = np.array(times) * 1000.0
    return {
        "throughput_mean": total_images / elapsed,
        "latency_mean_ms": elapsed / (total_images / batch_size) * 1000.0,
        "latency_p50_ms": float(np.percentile(times_ms, 50)),
        "latency_p95_ms": float(np.percentile(times_ms, 95)),
        "latency_p99_ms": float(np.percentile(times_ms, 99)),
        "peak_memory_mb": float(torch.cuda.max_memory_allocated() / 1e6),
    }


def _cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch._dynamo.reset()


def _gpu_slug() -> str:
    name = torch.cuda.get_device_name(0)
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()


# ---------------------------------------------------------------------------
# Variant builders — each returns (fwd_callable, model_for_cleanup)
# ---------------------------------------------------------------------------


def _cast(model: nn.Module, precision: str) -> nn.Module:
    if precision == "fp16":
        return model.half()
    if precision == "bf16":
        return model.bfloat16()
    return model.float()


def _input_dtype(precision: str) -> torch.dtype:
    return {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "amp": torch.float32,
    }[precision]


def build_reference(precision: str, device: torch.device, batch_size: int):
    ref = _seeded_wrapper(device, REF_SEED)
    ref = _cast(ref, precision).eval()
    static_in = torch.randn(batch_size, 12, 128, 128, device=device, dtype=_input_dtype(precision))
    if precision == "amp":
        def fwd(x: torch.Tensor | None = None):
            with torch.no_grad(), torch.amp.autocast("cuda"):
                return ref(static_in if x is None else x)
    else:
        def fwd(x: torch.Tensor | None = None):
            with torch.no_grad():
                return ref(static_in if x is None else x)
    return fwd, ref


def build_reference_patched(precision: str, device: torch.device, batch_size: int):
    """Reference + S2-specialized graph-break patches (eager)."""
    from optim_olmoearth.patches import apply_s2_specialized_patches
    apply_s2_specialized_patches()
    return build_reference(precision, device, batch_size)


def build_reference_patched_compile(
    precision: str, device: torch.device, batch_size: int, mode: str = "default"
):
    """Reference + patches + torch.compile(fullgraph=True)."""
    from optim_olmoearth.patches import apply_s2_specialized_patches
    apply_s2_specialized_patches()
    fwd, ref = build_reference(precision, device, batch_size)
    ref.encoder = torch.compile(ref.encoder, mode=mode, dynamic=False, fullgraph=True)
    return fwd, ref


def build_reference_compiled(precision: str, device: torch.device, batch_size: int, mode: str):
    fwd, ref = build_reference(precision, device, batch_size)
    # Compile only the inner encoder, since OlmoEarthWrapper.forward does
    # Python tensor construction that confuses dynamo.
    ref.encoder = torch.compile(ref.encoder, mode=mode, dynamic=False)
    return fwd, ref


def build_fast(precision: str, device: torch.device, batch_size: int):
    ref = _seeded_wrapper(device, REF_SEED).float().eval()
    fast = FastOlmoEarthBase.from_reference(ref).eval()
    del ref
    fast = _cast(fast, precision)
    static_in = torch.randn(batch_size, 12, 128, 128, device=device, dtype=_input_dtype(precision))
    if precision == "amp":
        def fwd(x: torch.Tensor | None = None):
            with torch.no_grad(), torch.amp.autocast("cuda"):
                return fast(static_in if x is None else x)
    else:
        def fwd(x: torch.Tensor | None = None):
            with torch.no_grad():
                return fast(static_in if x is None else x)
    return fwd, fast


def build_fast_compiled(precision: str, device: torch.device, batch_size: int, mode: str):
    ref = _seeded_wrapper(device, REF_SEED).float().eval()
    fast_inner = FastOlmoEarthBase.from_reference(ref).eval()
    del ref
    fast_inner = _cast(fast_inner, precision).eval()
    fast = torch.compile(fast_inner, mode=mode, dynamic=False, fullgraph=True)
    static_in = torch.randn(batch_size, 12, 128, 128, device=device, dtype=_input_dtype(precision))
    if precision == "amp":
        def fwd(x: torch.Tensor | None = None):
            with torch.no_grad(), torch.amp.autocast("cuda"):
                return fast(static_in if x is None else x)
    else:
        def fwd(x: torch.Tensor | None = None):
            with torch.no_grad():
                return fast(static_in if x is None else x)
    return fwd, fast


def build_fast_cuda_graph(precision: str, device: torch.device, batch_size: int):
    """Capture the fast wrapper into a CUDA Graph for fixed-shape replay."""
    ref = _seeded_wrapper(device, REF_SEED).float().eval()
    fast = FastOlmoEarthBase.from_reference(ref).eval()
    del ref
    fast = _cast(fast, precision).eval()

    static_in = torch.randn(
        batch_size, 12, 128, 128, device=device, dtype=_input_dtype(precision)
    )

    # Warm up on a side stream
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            with torch.no_grad():
                _ = fast(static_in)
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        with torch.no_grad():
            static_out = fast(static_in)

    def fwd(x: torch.Tensor | None = None):
        if x is not None:
            static_in.copy_(x)
        g.replay()
        # Return a clone — without it, callers reading static_out before the
        # next replay see correct data, but for safety we copy on equiv check.
        return static_out

    return fwd, fast


def build_fast_cuda_graph_compile(precision: str, device: torch.device, batch_size: int):
    """torch.compile(default) + CUDA Graphs capture (manual)."""
    ref = _seeded_wrapper(device, REF_SEED).float().eval()
    fast_inner = FastOlmoEarthBase.from_reference(ref).eval()
    del ref
    fast_inner = _cast(fast_inner, precision).eval()
    fast = torch.compile(fast_inner, mode="default", dynamic=False, fullgraph=True)

    static_in = torch.randn(
        batch_size, 12, 128, 128, device=device, dtype=_input_dtype(precision)
    )

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5):
            with torch.no_grad():
                _ = fast(static_in)
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        with torch.no_grad():
            static_out = fast(static_in)

    def fwd(x: torch.Tensor | None = None):
        if x is not None:
            static_in.copy_(x)
        g.replay()
        return static_out

    return fwd, fast


def build_onnxruntime(precision: str, device: torch.device, batch_size: int):
    """Export the fast wrapper to ONNX and run via onnxruntime-gpu (CUDA EP)."""
    import tempfile

    import onnxruntime as ort

    if precision == "amp":
        raise NotImplementedError("ORT path doesn't support amp toggling here")

    ref = _seeded_wrapper(device, REF_SEED).float().eval()
    fast_inner = FastOlmoEarthBase.from_reference(ref).eval()
    del ref
    fast_inner = _cast(fast_inner, precision).eval()

    in_dtype = _input_dtype(precision)
    dummy = torch.randn(batch_size, 12, 128, 128, device=device, dtype=in_dtype)

    onnx_dir = Path(tempfile.mkdtemp(prefix="oe_onnx_"))
    onnx_path = onnx_dir / f"fast_oe_{precision}_b{batch_size}.onnx"

    # Export with the model on the right device/dtype, keep static shape
    torch.onnx.export(
        fast_inner,
        (dummy,),
        str(onnx_path),
        input_names=["x"],
        output_names=["y"],
        opset_version=17,
        dynamic_axes=None,
    )

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = [
        ("CUDAExecutionProvider", {"device_id": device.index or 0}),
        "CPUExecutionProvider",
    ]
    sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=providers)

    # Pre-allocate IO bindings to keep memory transfers efficient
    static_in = torch.randn(batch_size, 12, 128, 128, device=device, dtype=in_dtype)
    out_shape = (batch_size, FastOlmoEarthBase.EMBED_DIM)
    static_out = torch.empty(out_shape, device=device, dtype=in_dtype)

    np_dtype = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}[precision]

    def fwd(x: torch.Tensor | None = None):
        if x is not None:
            static_in.copy_(x)
        io = sess.io_binding()
        io.bind_input(
            "x",
            "cuda",
            device.index or 0,
            np_dtype,
            tuple(static_in.shape),
            static_in.data_ptr(),
        )
        io.bind_output(
            "y",
            "cuda",
            device.index or 0,
            np_dtype,
            tuple(static_out.shape),
            static_out.data_ptr(),
        )
        sess.run_with_iobinding(io)
        return static_out

    return fwd, sess


# ---------------------------------------------------------------------------
# Equivalence wrapper
# ---------------------------------------------------------------------------


@torch.no_grad()
def _verify_variant(
    label: str,
    build_fn,
    precision: str,
    device: torch.device,
    raise_on_fail: bool = True,
) -> bool:
    """Build the variant on a small batch with seeded input and assert equivalence."""
    x = _seeded_input(device, _input_dtype(precision), REF_SEED)

    ref = _seeded_wrapper(device, REF_SEED)
    ref_cast = _cast(ref, precision).eval()
    if precision == "amp":
        with torch.amp.autocast("cuda"):
            ref_y = ref_cast(x)
    else:
        ref_y = ref_cast(x)
    torch.cuda.synchronize()
    ref_y = ref_y.detach().clone()
    del ref, ref_cast
    _cleanup()

    fwd, model = build_fn(precision, device, x.shape[0])
    y = fwd(x).detach().clone()
    torch.cuda.synchronize()
    ok = check_equivalence(y, ref_y, precision, label=label, raise_on_fail=raise_on_fail)
    del fwd, model, y, ref_y, x
    _cleanup()
    return ok


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


VARIANTS: dict[str, dict] = {
    "reference": {
        "build": build_reference,
        "compile_mode": "none",
    },
    "reference_compile_def": {
        "build": lambda p, d, b: build_reference_compiled(p, d, b, "default"),
        "compile_mode": "default",
    },
    "reference_compile_max": {
        "build": lambda p, d, b: build_reference_compiled(p, d, b, "max-autotune"),
        "compile_mode": "max-autotune",
    },
    "reference_patched": {
        "build": build_reference_patched,
        "compile_mode": "none",
    },
    "reference_patched_compile_def": {
        "build": lambda p, d, b: build_reference_patched_compile(p, d, b, "default"),
        "compile_mode": "default+fullgraph",
    },
    "reference_patched_compile_reduce": {
        "build": lambda p, d, b: build_reference_patched_compile(p, d, b, "reduce-overhead"),
        "compile_mode": "reduce-overhead+fullgraph",
    },
    "fast": {
        "build": build_fast,
        "compile_mode": "none",
    },
    "fast_compile_def": {
        "build": lambda p, d, b: build_fast_compiled(p, d, b, "default"),
        "compile_mode": "default",
    },
    "fast_compile_max": {
        "build": lambda p, d, b: build_fast_compiled(p, d, b, "max-autotune"),
        "compile_mode": "max-autotune",
    },
    "fast_compile_max_nocg": {
        "build": lambda p, d, b: build_fast_compiled(p, d, b, "max-autotune-no-cudagraphs"),
        "compile_mode": "max-autotune-no-cudagraphs",
    },
    "fast_compile_reduce": {
        "build": lambda p, d, b: build_fast_compiled(p, d, b, "reduce-overhead"),
        "compile_mode": "reduce-overhead",
    },
    "fast_cuda_graph": {
        "build": build_fast_cuda_graph,
        "compile_mode": "cuda-graph",
    },
    "fast_cuda_graph_compile": {
        "build": build_fast_cuda_graph_compile,
        "compile_mode": "cuda-graph+compile",
    },
    "onnxruntime": {
        "build": build_onnxruntime,
        "compile_mode": "onnx",
    },
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument(
        "--precisions",
        nargs="+",
        default=["fp32", "fp16", "amp"],
        choices=["fp32", "fp16", "bf16", "amp"],
    )
    p.add_argument(
        "--variants",
        nargs="+",
        default=list(VARIANTS.keys()),
        choices=list(VARIANTS.keys()),
    )
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--timed-seconds", type=float, default=30.0)
    p.add_argument("--output", default=None)
    p.add_argument(
        "--skip-equiv",
        action="store_true",
        help="Skip equivalence checks (faster; only do this if you've already validated).",
    )
    args = p.parse_args()

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    torch.set_float32_matmul_precision("high")

    out_path = Path(args.output) if args.output else (
        Path(__file__).parent / "results" / f"{_gpu_slug()}_optim.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = out_path.exists() and out_path.stat().st_size > 0
    csv_file = open(out_path, "a", newline="")
    fieldnames = [
        "variant",
        "precision",
        "compile_mode",
        "batch_size",
        "throughput_mean",
        "latency_mean_ms",
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_p99_ms",
        "peak_memory_mb",
        "equiv_ok",
        "gpu_name",
        "gpu_mem_gb",
        "pytorch_version",
        "timestamp",
    ]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()

    print(f"🖥  Device: {gpu_name} ({gpu_mem_gb:.1f} GB)")
    print(f"🎯 batch_size={args.batch_size}  warmup={args.warmup}  timed={args.timed_seconds}s")
    print(f"💾 output={out_path}\n")

    for variant_name in args.variants:
        variant = VARIANTS[variant_name]
        for precision in args.precisions:
            print(f"--- {variant_name} | {precision}")

            # Equivalence check on small batch
            equiv_ok = True
            if not args.skip_equiv:
                try:
                    equiv_ok = _verify_variant(
                        variant_name, variant["build"], precision, device,
                        raise_on_fail=False,
                    )
                except Exception as e:
                    print(f"  ⚠ equivalence check raised: {type(e).__name__}: {e}")
                    equiv_ok = False
                _cleanup()

            # Build fresh for benchmark
            try:
                fwd, model = variant["build"](precision, device, args.batch_size)
            except torch.cuda.OutOfMemoryError:
                print("  OOM at build")
                writer.writerow({
                    "variant": variant_name, "precision": precision,
                    "compile_mode": variant["compile_mode"], "batch_size": args.batch_size,
                    "throughput_mean": "OOM", "latency_mean_ms": "", "latency_p50_ms": "",
                    "latency_p95_ms": "", "latency_p99_ms": "", "peak_memory_mb": "",
                    "equiv_ok": equiv_ok, "gpu_name": gpu_name,
                    "gpu_mem_gb": f"{gpu_mem_gb:.1f}",
                    "pytorch_version": torch.__version__,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                })
                csv_file.flush()
                _cleanup()
                continue

            # All fwd functions accept an optional override; for the timer we
            # call with no argument so they use the static input.
            _fwd_local = fwd

            def fwd_no_arg():
                return _fwd_local(None)

            try:
                stats = _time_callable(
                    fwd_no_arg, args.batch_size, args.warmup, args.timed_seconds,
                )
                tput = stats["throughput_mean"]
                print(f"  → {tput:.1f} img/s  | mean {stats['latency_mean_ms']:.2f} ms  | "
                      f"peak {stats['peak_memory_mb']:.0f} MB  | equiv={equiv_ok}")
                writer.writerow({
                    "variant": variant_name,
                    "precision": precision,
                    "compile_mode": variant["compile_mode"],
                    "batch_size": args.batch_size,
                    "throughput_mean": f"{tput:.2f}",
                    "latency_mean_ms": f"{stats['latency_mean_ms']:.3f}",
                    "latency_p50_ms": f"{stats['latency_p50_ms']:.3f}",
                    "latency_p95_ms": f"{stats['latency_p95_ms']:.3f}",
                    "latency_p99_ms": f"{stats['latency_p99_ms']:.3f}",
                    "peak_memory_mb": f"{stats['peak_memory_mb']:.1f}",
                    "equiv_ok": equiv_ok,
                    "gpu_name": gpu_name,
                    "gpu_mem_gb": f"{gpu_mem_gb:.1f}",
                    "pytorch_version": torch.__version__,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                })
                csv_file.flush()
            except torch.cuda.OutOfMemoryError:
                print("  OOM during timing")
            except Exception as e:
                print(f"  ✗ failed: {type(e).__name__}: {e}")
            finally:
                del fwd, fwd_no_arg, model
                _cleanup()

    csv_file.close()
    print(f"\n✅ done → {out_path}")


if __name__ == "__main__":
    main()
