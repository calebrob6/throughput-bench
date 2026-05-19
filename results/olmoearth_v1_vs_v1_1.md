# OlmoEarth v1 vs v1.1 throughput comparison

Same input contract on both: `(B, 12, 128, 128)` Sentinel-2 L2A
batches. Hardware: Tesla V100-SXM2 32 GB, bs=512, 30 s timed, V100
SDPA falls back to cutlass mem-efficient (sm_70, no Flash-Attention).

## Architecture deltas

| | v1 | v1.1 |
|---|---|---|
| S2 L2A bandsets | **3** (`B02 B03 B04 B08`, `B05 B06 B07 B8A B11 B12`, `B01 B09`) | **1** (all 12 bands in one group) |
| Patch embedding | `Conv2d(in_chans=c_bs, out=768, k=8, s=8)` per bandset | `Linear(p*p*12 → patch_hidden → embed)` (one MLP for all 12 bands) |
| `patch_embed_hidden_sizes` | (absent) | `[12]` Nano, `[64]` Tiny/Base |
| `use_linear_patch_embed` | `False` (Conv2d) | `True` (cuBLAS GEMM) |
| Encoder hyperparams (embed / depth / heads) | unchanged | unchanged |

Both register the same 9 supported modalities (S2L2A, S1, Landsat,
WorldCover, SRTM, OSM-raster, WRI-canopy-height, CDL, WorldCereal) and
expect the same wrapper input shape. **The single-bandset switch is the
critical change**: where v1 produced `16×16×3 = 768` S2 tokens per
sample for the transformer, v1.1 produces only `16×16×1 = 256` tokens.

## Per-sample compute

| size | v1 params (M) | v1.1 params (M) | v1 MACs (G) | v1.1 MACs (G) | MAC ratio |
|------|--------------:|----------------:|------------:|--------------:|----------:|
| nano | 1.4 | **1.7** (+21 %) | 1.26 | **0.46** | **2.74× less** |
| tiny | 6.2 | **12.5** (+101 %) | 8.23 | **3.15** | **2.61× less** |
| base | 89.0 | **114.0** (+28 %) | 130.76 | **45.12** | **2.90× less** |

v1.1 has *more parameters* (patch-embed MLP + extra patch-hidden
dimension) but **far less compute per sample** because the transformer
sees 1/3 the tokens. The MAC ratio (~2.7×) is most of the throughput
story; the residual ~25 % speedup comes from the Linear-vs-Conv2d
patch embed being faster on V100 cuDNN.

## Throughput (img/s) — V100, bs=512

| size | prec | compile | v1 | **v1.1** | v1.1/v1 |
|------|------|---------|-----:|---------:|--------:|
| nano | fp32 | none    |  1,909.6 |    7,660.1 | **4.01×** |
| nano | fp32 | default |  2,092.7 |    9,799.6 | **4.68×** |
| nano | fp16 | none    |  4,932.8 |   18,915.7 | **3.83×** |
| nano | fp16 | default |  5,345.8 |   25,768.2 | **4.82×** |
| nano | amp  | none    |  4,367.5 |   14,879.1 | **3.41×** |
| nano | amp  | default |  5,286.8 |   25,048.5 | **4.74×** |
| tiny | fp32 | none    |    626.5 |    1,692.2 | **2.70×** |
| tiny | fp32 | default |    665.4 |    2,240.4 | **3.37×** |
| tiny | fp16 | none    |  2,112.1 |    6,702.6 | **3.17×** |
| tiny | fp16 | default |  2,350.2 |    8,380.4 | **3.57×** |
| tiny | amp  | none    |  1,764.9 |    5,349.7 | **3.03×** |
| tiny | amp  | default |  2,293.6 |    8,122.8 | **3.54×** |
| base | fp32 | none    |     78.0 |      262.7 | **3.37×** |
| base | fp32 | default |     83.4 |      276.3 | **3.31×** |
| base | fp16 | none    |    354.0 |    1,220.6 | **3.45×** |
| base | fp16 | default |    360.8 |    1,284.2 | **3.56×** |
| base | amp  | none    |    306.3 |    1,047.0 | **3.42×** |
| base | amp  | default |    353.7 |    1,267.6 | **3.58×** |

**v1.1 is uniformly 2.7–4.8× faster than v1**, with the gap widest at
Nano (where the per-bandset Conv2d patch embed was disproportionately
expensive relative to the tiny transformer).

## Peak GPU memory (MB) — V100, bs=512

| size | prec | compile | v1 | v1.1 | v1.1/v1 |
|------|------|---------|-----:|-----:|--------:|
| nano | fp16 | default |  1,450 |    668 | **0.46×** |
| tiny | fp16 | default |  2,074 |  2,394 | 1.15× |
| base | fp16 | default |  7,927 |  2,882 | **0.36×** |
| base | fp32 | none    | 18,160 |  6,517 | **0.36×** |

Activation memory drops dramatically for Nano and Base (one bandset
means 1/3 the activations at every transformer layer). Tiny is a wash
because the encoder MLP and embed dim adjustments add activation cost
that roughly cancels the bandset savings.

## Caveats

* Pretrained `v1` runs were collected 2026-04-29 with the same
  `pytorch_version=2.10.0+cu128`; v1.1 was collected 2026-05-19 on the
  same V100 host. Same driver / clocks per `results/tesla_v100_sxm2_32gb_hardware.json`.
* V100 sm_70 SDPA is cutlass mem-efficient for both versions
  (no Flash-Attention). The v1.1 speedup is **not** attention-kernel
  related — it's all token-count + patch-embed.
* Throughput is for the pre-allocated GPU batch path (the default in
  `benchmark.py`). DataLoader-path numbers should track these ratios
  closely because input prep cost is identical (same shape, same
  spectral stats).

## Reproducing this comparison

The raw CSV is `results/tesla_v100_sxm2_32gb.csv`. To rebuild the
tables in this doc:

```bash
python -c "
import csv
rows = list(csv.DictReader(open('results/tesla_v100_sxm2_32gb.csv')))
def get(n, p, c, bs='512'):
    return next((r for r in rows
                 if r['model_name']==n and r['precision']==p
                    and r['compile_mode']==c and r['batch_size']==bs), None)
for size in ['nano', 'tiny', 'base']:
    for prec in ['fp32', 'fp16', 'amp']:
        for cm in ['none', 'default']:
            a, b = get(f'olmoearth_{size}', prec, cm), get(f'olmoearth_v1_1_{size}', prec, cm)
            if a and b:
                print(size, prec, cm, float(a['throughput_mean']), float(b['throughput_mean']))
"
```
