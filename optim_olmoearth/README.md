# OlmoEarth-Base Inference Optimization

Drop-in replacements for `geo_models.OlmoEarthWrapper("base")` that produce
**bit-equivalent outputs** (within standard fp16/bf16 rounding) but run
faster and/or use less memory. Targets the wrapper's input contract:

* Input  — `(B, 12, 128, 128)` float on CUDA (Sentinel-2 L2A, 12 bands,
  single timestep)
* Output — `(B, 768)` spatially-pooled S2 token embedding

## TL;DR results — Tesla V100-SXM2 32 GB, batch 512

| precision | variant                              | img/s | speedup | peak mem |
|-----------|--------------------------------------|-------|---------|----------|
| fp32      | `reference` (baseline)               |  79.0 |  1.00x  | 17.7 GB  |
| fp32      | `reference_patched` + `compile`      |  84.8 |  1.07x  | 11.7 GB  |
| fp32      | `reference_patched` + `compile-reduce`|  84.6 |  1.07x  | **0.79 GB** |
| fp32      | `fast` + `compile`                   |  83.7 |  1.06x  |  9.2 GB  |
| **fp16**  | **`reference` (baseline)**           | **352.8** | **1.00x** | **8.9 GB** |
| fp16      | `reference_patched` + `compile`      | 366.2 |  1.04x  |  5.9 GB  |
| fp16      | `fast`                               | 356.8 |  1.01x  |  8.9 GB  |
| fp16      | `fast` + `compile`                   | **369.1** | **1.05x** | 4.6 GB |
| fp16      | `fast` + `compile-reduce` (CG fused) | 366.3 |  1.04x  | **0.39 GB** |
| fp16      | `fast` + manual CUDA Graph           | 356.6 |  1.01x  | **0.44 GB** |
| amp       | `reference` (baseline)               | 310.0 |  1.00x  | 10.0 GB  |
| amp       | `fast` + `compile`                   | 362.4 |  1.17x  |  5.6 GB  |

Best throughput: **`fast_compile_def` at 369 img/s fp16 (+5%)** plus
**`fast_compile_reduce` at 366 img/s with 23x lower peak memory (387 MB
vs 8.9 GB)** — that headroom enables much larger concurrent batches in
real serving.

The reference + graph-break patches + compile reaches essentially the
same throughput (366 img/s) without rewriting the model, just by
removing 3 graph breaks in the upstream `flexi_vit` code.

## What's in this directory

```
optim_olmoearth/
├── equiv.py              tolerance helpers + check_equivalence()
├── fast_wrapper.py       FastOlmoEarthBase — fullgraph-compatible rewrite
├── patches.py            monkey-patches that remove upstream graph breaks
├── reference_set.py      build a deterministic reference input/output set
├── bench_optim.py        benchmark + equivalence runner
├── tests/
│   └── test_fast_wrapper.py   pytest-style equivalence smoke test
└── results/              CSV outputs per GPU
```

## Optimization techniques applied

### 1. Eliminating Python overhead in `OlmoEarthWrapper.forward`

The reference allocates a `MaskedOlmoEarthSample` dataclass + 5-D
all-zeros mask on every forward, then immediately throws away most of
its fields inside the encoder. Our `FastOlmoEarthBase.forward` skips
all of it and feeds tokens directly into the transformer.

### 2. Fused QKV + LayerNorm

The reference uses 3 separate `nn.Linear` for q/k/v in every attention
block; we fuse to one `nn.Linear(dim, 3*dim)`. The original Linears'
weights are stitched into the fused weight at construction time. Saves
two kernel launches per block × 12 blocks × 2 ms ≈ ~50 ms of cumulative
launch overhead per bs=512 batch on V100.

### 3. Linear patch embedding instead of Conv2d

`FlexiPatchEmbed` ships with both a Conv2d path (default — needed for
loading old checkpoints) and a Linear path that hits cuBLAS GEMM. The
linear path is faster on V100 for small in_chans (the reference Conv2d
is `(out=768, in=2..6, k=8, s=8)` — cuDNN's 1×1-style paths are
inefficient at that shape). We absorb the 3 reference Conv2d weights
into a single `nn.Linear(12*64 → 3*768)` that handles all bandsets in
one matmul.

### 4. Pre-baked composite encoding

`CompositeEncodings.forward` recomputes channel/time/month/spatial
encodings on every call. For our fixed input shape and `T=1, mask=0,
patch_size=8, input_res=10`, the additive bias is a *constant* tensor
of shape `(1, 768, 768)`. We cache it once at module construction and
add it as a buffer.

### 5. Eliminating `torch.compile` graph breaks (`patches.py`)

Profiling the reference under `torch.compile` reveals 4 dynamo subgraphs
(3 graph breaks):

* `flexi_vit.py:485, :906, :912` — `logger.debug(f"...")` calls.
* `flexi_vit.py:43` — `set(available_modalities).intersection(set(...))`.
* `MaskedOlmoEarthSample.modalities` — NamedTuple property iterating
  `self._fields` and calling `getattr(self, x) is not None` per field.

`apply_safe_patches()` fixes the first two without changing semantics.
`apply_s2_specialized_patches()` additionally replaces
`MultiModalPatchEmbeddings.forward` and `Encoder.apply_attn` with
versions that hardcode the S2-only fast-pass path — together they make
the reference encoder **fullgraph-compatible** (1 subgraph, no breaks).

After patching:
* `reference_patched_compile_def` runs at fast-wrapper speeds (366 img/s
  fp16) without any model rewrite.
* `reference_patched_compile_reduce` adds CUDA-graph capture for
  another 23x memory reduction.

### 6. CUDA Graphs

`fast_cuda_graph` captures the entire fast wrapper into a `cuda.CUDAGraph`
for fixed-shape replay. Throughput is the same (compute-bound at
bs=512) but memory drops dramatically because activations are reused.

`torch.compile(mode="reduce-overhead")` does this automatically and is
the recommended path; `fast_cuda_graph_compile` combines compile +
manual CUDA Graph for the same effect.

### 7. ONNX Runtime

`build_onnxruntime` exports the fast wrapper to ONNX and runs via
`onnxruntime-gpu` with IO bindings. On V100 the ORT CUDA EP is slower
than PyTorch's mem-efficient SDPA path, so this variant doesn't win
here, but the export works and is bit-equivalent — useful for serving
backends that require ONNX.

## Why no fp16 Flash-Attention?

V100 (sm_70) is below Flash-Attention 2's minimum (sm_80). PyTorch SDPA
falls back to the cutlass mem-efficient kernel
(`fmha_cutlassF_f16_aligned_64x64_rf_sm70`), which is what every
variant ends up using. Without sm_80+ Tensor Cores there's no faster
attention path on this GPU.

The same code on H100/A100 would automatically pick the Flash-Attention
backend and run substantially faster — re-running this benchmark on
those GPUs is the highest-leverage next step.

## Reproducing

```bash
# 1. Build the reference set (one-time, ~360 MB cached fixture)
python -m optim_olmoearth.reference_set --device cuda:0

# 2. Smoke-test that FastOlmoEarthBase matches the reference
python optim_olmoearth/tests/test_fast_wrapper.py --device cuda:0

# 3. Run all variants at all precisions on a free GPU
python optim_olmoearth/bench_optim.py \
    --gpu-id 0 --batch-size 512 --timed-seconds 30 \
    --precisions fp32 fp16 amp \
    --variants reference reference_patched_compile_def \
               fast fast_compile_def fast_compile_reduce \
               fast_cuda_graph_compile
```

Results land in
`optim_olmoearth/results/<gpu_slug>_optim.csv`.

## Notes on equivalence

* **fp32**: bit-exact within ~1e-7 (matmul/SDPA reduction order).
* **fp16**: ~2e-3 max-abs over a 768-dim output; well within the noise
  of any downstream linear probe.
* **AMP**: ~5e-3 max-abs, ~1.3e-3 mean-abs. Slightly looser tolerance
  is needed because `torch.amp.autocast` picks fp16 for matmul and
  fp32 for accumulators per-op, and the per-op picks differ when the
  graph is restructured.
