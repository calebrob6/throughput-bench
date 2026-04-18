# TODO — critical review action items

## Bugs (fix first)

- [ ] **Segmentation `macs_G` is wrong.** `run_single_benchmark` in `benchmark.py:449–452` recomputes `task_params` on the wrapped SMP U-Net but **not** `task_macs`. Every segmentation row reports the classification-backbone MACs (verified: ResNet-50 seg row shows `macs_G=8.17`, same as classification). CLAUDE.md claims otherwise — either fix the code or fix the doc.
- [ ] **`throughput_std / median / min / max` are fake.** `_format_gpu_stats` in `benchmark.py:402` sets all four equal to the single aggregate `throughput_mean`. Drop the columns to stop implying noise was measured. Remove the columns from the current results to avoid confusion.
- [ ] **Doc says CUDA events are used for timing; code uses `time.perf_counter()`** with `cuda.synchronize()` (`benchmark_gpu`, `benchmark_gpu_preallocated`). Reconcile docstring/README with reality (or switch to `torch.cuda.Event` for sub-ms resolution).
- [ ] **H100 fp32 classification has 28/29 models, not 29.** Silent missing row — investigate auto-BS / OOM path and add a log line when a config is skipped without writing a row.

## Measurement protocol gaps

- [ ] **Fix cross-precision BS parity.** `find_max_batch_size` (`benchmark.py:229`) probes at fp32 uncompiled, then reuses that BS for fp16/amp — under-reports fp16 throughput because fp16 could fit ~2× the batch. Probe per precision (do this smartly by starting at the previous BS).
- [ ] **Stop labeling TF32 as `fp32`.** `torch.set_float32_matmul_precision("high")` (`benchmark.py:557`) enables TF32 Tensor Cores on Ampere+, so "fp32" on H100/RTX 6000 is TF32 while V100 runs IEEE-754. Add a `tf32_enabled` column or rename the precision to `tf32` on capable GPUs.
- [ ] **Add `torch.compile` runs.** Currently every committed row is `compile_mode=none`. Add `default` and `max-autotune` for at least H100/RTX 6000/Ada. This is the single biggest missing story for modern GPUs.
- [ ] **Add `bf16`.** Standard training precision on Ampere+/Hopper; absent today. Skip on V100 (unsupported).
- [ ] **Capture p50/p95/p99 latency.** `latency_mean_ms` is `elapsed / num_iterations`. Record per-batch times during the timed window and emit percentiles.

## Coverage — the repo markets broader than it measures

- [ ] **Vary input channels and spatial size.** Fixed 3×224×224 isn't representative of EO work. Add small sweep for channels (3/4/6/13).
- [ ] **Find a decoder that all backbones can use** Currently single SMP U-Net only that doesn't work for the ViTs.


## Reproducibility metadata

- [ ] **Extend `hardware.json`** to include: NVIDIA driver version (`nvidia-smi --query-gpu=driver_version`), persistence mode, power cap, clock state, topology, and the git SHA of the benchmark run. Driver/clock alone can move H100 throughput ±15%.
