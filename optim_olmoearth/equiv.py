"""Tolerance helpers for validating optimized OlmoEarth variants.

Each precision has its own tolerance (max-abs and mean-abs) that we expect
of any equivalent optimization (refactor, fused kernels, attention impl
swap, compile, CUDA Graphs, ONNX Runtime, etc.). Anything tighter than
that — bit-exact rewrites, no-op refactors — is rare in float arithmetic
because reduction order, kernel selection, and Tensor Core promotion all
introduce small rounding differences.

Tolerances are deliberately generous on max-abs (cumulative across the
12-layer attention path) but reasonably tight on mean-abs.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Tolerance:
    atol_max: float
    atol_mean: float
    rtol_p99: float


# Tuned empirically: fp16/bf16 attention layers compound rounding error
# across 12 layers and ~336 tokens; mean abs stays small but max-abs can
# drift a few percent of the typical activation magnitude.
TOLERANCES: dict[str, Tolerance] = {
    "fp32": Tolerance(atol_max=5e-4, atol_mean=5e-6, rtol_p99=1e-3),
    "fp16": Tolerance(atol_max=8e-2, atol_mean=2e-3, rtol_p99=5e-2),
    "bf16": Tolerance(atol_max=2e-1, atol_mean=5e-3, rtol_p99=1e-1),
    # AMP picks fp16 vs fp32 per-op; small reductions in different orders
    # accumulate slightly differently between equivalent graphs. We've
    # observed ~1% mean relative error and ~10% p99 even between
    # mathematically-identical AMP forwards.
    "amp":  Tolerance(atol_max=8e-2, atol_mean=3e-3, rtol_p99=1.5e-1),
}


def diff_stats(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    """Compute element-wise diff stats in float64 for stability."""
    a = actual.detach().to(torch.float64).flatten()
    e = expected.detach().to(torch.float64).flatten()
    diff = (a - e).abs()
    rel = diff / (e.abs() + 1e-8)
    return {
        "abs_max":  float(diff.max().item()),
        "abs_mean": float(diff.mean().item()),
        "rel_p99":  float(rel.quantile(0.99).item()),
        "ref_mean": float(e.mean().item()),
        "ref_std":  float(e.std().item()),
    }


def check_equivalence(
    actual: torch.Tensor,
    expected: torch.Tensor,
    precision: str,
    label: str = "<variant>",
    tol: Tolerance | None = None,
    raise_on_fail: bool = True,
) -> bool:
    """Compare ``actual`` against ``expected`` under the given precision tolerance."""
    tol = tol or TOLERANCES[precision]
    s = diff_stats(actual, expected)
    ok = (
        s["abs_max"]  <= tol.atol_max
        and s["abs_mean"] <= tol.atol_mean
        and s["rel_p99"] <= tol.rtol_p99
    )
    status = "OK " if ok else "FAIL"
    print(
        f"  [{status}] {label:40s} {precision:>4s}: "
        f"abs_max={s['abs_max']:.3e} (≤{tol.atol_max:.0e})  "
        f"abs_mean={s['abs_mean']:.3e} (≤{tol.atol_mean:.0e})  "
        f"rel_p99={s['rel_p99']:.3e} (≤{tol.rtol_p99:.0e})"
    )
    if raise_on_fail and not ok:
        raise AssertionError(
            f"{label} ({precision}) failed equivalence: {s} vs tol {tol}"
        )
    return ok
