"""Validate that ``FastOlmoEarthBase`` matches the reference output."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from optim_olmoearth.equiv import check_equivalence  # noqa: E402
from optim_olmoearth.fast_wrapper import FastOlmoEarthBase  # noqa: E402
from optim_olmoearth.reference_set import (  # noqa: E402
    REF_SEED,
    _seeded_input,
    _seeded_wrapper,
)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--precisions", nargs="+", default=["fp32", "fp16", "bf16"])
    args = p.parse_args()
    device = torch.device(args.device)

    for precision in args.precisions:
        if precision == "fp32":
            dtype = torch.float32
            cast = lambda m: m.float()  # noqa: E731
        elif precision == "fp16":
            dtype = torch.float16
            cast = lambda m: m.half()  # noqa: E731
        else:
            dtype = torch.bfloat16
            cast = lambda m: m.bfloat16()  # noqa: E731

        ref = _seeded_wrapper(device, REF_SEED)
        ref = cast(ref).eval()
        x = _seeded_input(device, dtype, REF_SEED)

        ref_y = ref(x)
        torch.cuda.synchronize(device)

        fast = FastOlmoEarthBase.from_reference(ref).eval()
        fast_y = fast(x)
        torch.cuda.synchronize(device)

        check_equivalence(fast_y, ref_y, precision, label="FastOlmoEarthBase")

        del ref, fast, ref_y, fast_y, x
        torch.cuda.empty_cache()

    print("\nAll precisions passed equivalence checks.")


if __name__ == "__main__":
    main()
