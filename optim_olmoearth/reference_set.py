"""Generate and load reference inputs/outputs for OlmoEarth-Base.

Uses the canonical ``geo_models.OlmoEarthWrapper`` as the source of truth.
All optimized variants are validated against these tensors with
``check_equivalence`` from ``optim_olmoearth.equiv``.

The reference set is generated once on the GPU you intend to benchmark on
(weights are random — see note below — so the reference is reproducible
only when seeded the same way and run on the same hardware/dtype).

Note on random weights
----------------------
``OlmoEarthWrapper`` constructs the model with random weights (no
pretrained checkpoint download). For a fixed seed, every Python process
that imports the wrapper produces the same weights, so the reference
output is reproducible across runs *of the same Python interpreter on the
same hardware*. We snapshot the random weights into a state_dict and
reload them in optimization variants so weight identity is guaranteed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

REF_DIR = Path(__file__).parent / "fixtures"
REF_PATH = REF_DIR / "reference_io.pt"
REF_BATCH = 4
REF_INPUT_SHAPE = (REF_BATCH, 12, 128, 128)
REF_SEED = 0xC0FFEE


def _seeded_wrapper(device: torch.device, seed: int = REF_SEED):
    """Build an ``OlmoEarthWrapper`` with deterministic random weights."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from geo_models import OlmoEarthWrapper

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    wrapper = OlmoEarthWrapper(model_size="base").to(device).eval()
    return wrapper


def _seeded_input(device: torch.device, dtype: torch.dtype, seed: int = REF_SEED) -> torch.Tensor:
    """Build a deterministic ``(B, 12, 128, 128)`` input tensor."""
    g = torch.Generator(device=device).manual_seed(seed + int(dtype.itemsize))
    return torch.randn(REF_INPUT_SHAPE, generator=g, device=device, dtype=dtype)


@torch.no_grad()
def generate(device: torch.device | str = "cuda:0", overwrite: bool = False) -> Path:
    """Generate the reference set and save to ``REF_PATH``.

    Saves a dict containing:
      * ``state_dict``        – encoder weights (so optim variants share params)
      * ``inputs[precision]`` – seeded input tensor in the right dtype
      * ``outputs[precision]``– reference output produced by ``OlmoEarthWrapper``

    Reference outputs are produced for fp32, fp16, bf16. AMP is not stored
    separately because its output equals fp32-with-autocast and tolerance
    is the same as fp16.
    """
    device = torch.device(device)
    REF_DIR.mkdir(parents=True, exist_ok=True)
    if REF_PATH.exists() and not overwrite:
        print(f"  reference already exists at {REF_PATH} (pass --overwrite to regen)")
        return REF_PATH

    wrapper = _seeded_wrapper(device)

    bundle: dict = {
        "input_shape": REF_INPUT_SHAPE,
        "seed": REF_SEED,
        "model_size": "base",
        "device_capability": tuple(torch.cuda.get_device_capability(device)),
        "device_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "state_dict": {k: v.detach().cpu().clone() for k, v in wrapper.state_dict().items()},
        "inputs": {},
        "outputs": {},
    }

    for precision in ("fp32", "fp16", "bf16"):
        if precision == "fp32":
            dtype = torch.float32
            cast = lambda m: m.float()  # noqa: E731
        elif precision == "fp16":
            dtype = torch.float16
            cast = lambda m: m.half()  # noqa: E731
        else:
            dtype = torch.bfloat16
            cast = lambda m: m.bfloat16()  # noqa: E731

        # Re-instantiate from the saved state_dict so we always cast from fp32
        ref_wrapper = _seeded_wrapper(device)
        ref_wrapper.load_state_dict(bundle["state_dict"])
        ref_wrapper = cast(ref_wrapper).eval()

        x = _seeded_input(device, dtype)
        y = ref_wrapper(x)
        torch.cuda.synchronize(device)

        bundle["inputs"][precision] = x.detach().cpu().clone()
        bundle["outputs"][precision] = y.detach().cpu().clone()
        print(
            f"  {precision:>4s}: out shape {tuple(y.shape)} "
            f"mean={y.float().mean().item():+.4e} std={y.float().std().item():.4e}"
        )

        del ref_wrapper, x, y
        torch.cuda.empty_cache()

    torch.save(bundle, REF_PATH)
    print(f"✅ wrote {REF_PATH} ({REF_PATH.stat().st_size / 1e6:.1f} MB)")
    return REF_PATH


def load(device: torch.device | str = "cuda:0") -> dict:
    """Load the reference bundle; tensors are moved to ``device``."""
    if not REF_PATH.exists():
        raise FileNotFoundError(
            f"No reference set at {REF_PATH}; run "
            "`python -m optim_olmoearth.reference_set` first."
        )
    bundle = torch.load(REF_PATH, map_location="cpu", weights_only=False)
    device = torch.device(device)
    bundle["inputs"] = {k: v.to(device) for k, v in bundle["inputs"].items()}
    bundle["outputs"] = {k: v.to(device) for k, v in bundle["outputs"].items()}
    return bundle


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate(args.device, overwrite=args.overwrite)
