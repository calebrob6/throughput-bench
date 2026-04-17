# Contributing Benchmark Results

We welcome benchmark results from different hardware! Here's how to contribute:

## Quick Start

```bash
# 1. Clone and set up
git clone https://github.com/calebrob6/geospeedy.git
cd geospeedy
make setup

# 2. Run the benchmark (uses GPU 0 by default)
make benchmark

# 3. Check your results
make visualize

# 4. Open a PR with your results
git checkout -b results/my-gpu-name
git add results/ figures/
git commit -m "Add benchmark results for <your GPU>"
git push origin results/my-gpu-name
```

## What Gets Generated

Running `make benchmark` creates two files in `results/`:

- **`{gpu_slug}.csv`** — Benchmark results (e.g., `a100_80gb.csv`)
- **`{gpu_slug}_hardware.json`** — Hardware metadata (GPU, CPU, PyTorch version, etc.)

The GPU slug is auto-detected from your hardware (e.g., `tesla_v100_sxm2_32gb`, `a100_sxm4_80gb`).

## Benchmark Parameters

The default `make benchmark` runs:

- **All 29 models** across 10 architecture families
- **Classification + segmentation** tasks
- **fp32, fp16, AMP** precision modes
- **Auto batch size**: Finds the largest power-of-2 batch size that fits in your GPU memory
- **20 warmup iterations**, then **30 seconds** of timed inference per config
- **Single GPU**, verified empty before benchmarking

## Custom Runs

```bash
# Use a specific GPU
make benchmark GPU_ID=2

# Quick test with a few models
make benchmark-quick

# Only classification
python benchmark.py --gpu-id 0 --tasks classification

# Specific models
python benchmark.py --gpu-id 0 --models resnet50 vit_large_patch16_224
```

## PR Checklist

- [ ] Results CSV and hardware JSON are in `results/`
- [ ] GPU was idle during benchmarking (no other processes)
- [ ] You used the default benchmark settings (`make benchmark`)
- [ ] Hardware JSON shows correct GPU info
