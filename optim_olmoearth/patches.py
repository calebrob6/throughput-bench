"""Runtime patches that remove ``torch.compile`` graph breaks from upstream
``olmoearth_pretrain_minimal``.

The reference encoder produces 4 dynamo subgraphs (3 graph breaks) on a
plain ``torch.compile`` call, which lowers compile gains because Inductor
can only fuse ops within a single graph.

Sources of the breaks (verified with ``TORCH_LOGS=graph_breaks``):
    1. ``logger.debug(f"...")`` f-string calls in
       * ``flexi_vit.py:485`` (``MultiModalPatchEmbeddings.apply_embedding_to_modality``)
       * ``flexi_vit.py:906`` and ``:912`` (``CompositeEncodings._apply_encodings_per_modality``)
    2. ``set(available_modalities).intersection(set(supported_modality_names))``
       in ``flexi_vit.get_modalities_to_process`` — Dynamo can't materialize
       Python ``set`` from non-symbolic strings.
    3. ``MaskedOlmoEarthSample.modalities`` (and ``TokensAndMasks.modalities``)
       NamedTuple property which iterates ``self._fields`` and calls
       ``getattr(self, field) is not None`` per field — Dynamo trips on
       ``hasattr`` calls against the NamedTuple wrapper.

``apply_safe_patches()`` fixes (1) and (2) without changing model
mathematics — every model that imports ``olmoearth_pretrain_minimal``
becomes lighter on graph breaks.

``apply_s2_specialized_patches()`` additionally fixes (3) by replacing
``MultiModalPatchEmbeddings.forward`` and ``Encoder.apply_attn`` with
versions that hardcode the S2-only fast-pass inference path. This keeps
identical semantics for our wrapper's input contract (S2 L2A only,
``fast_pass=True``, mask = all visible) but is unsafe for general use.
"""

from __future__ import annotations


class _NoLogger:
    """Replacement for ``logging.Logger`` that drops every message."""

    def debug(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def critical(self, *args, **kwargs):
        pass


_SAFE_APPLIED = False
_S2_APPLIED = False


def apply_safe_patches() -> None:
    """Idempotent monkey-patch that removes break sources (1) and (2).

    These changes are *transparent*: outputs are bit-identical to the
    unpatched code (the patched ``get_modalities_to_process`` returns a
    list in input order rather than a set in arbitrary order, but the
    model's downstream loops also iterate the result in order, so the
    final tensor stack is the same when only one modality is active —
    which is the case for our wrapper).
    """
    global _SAFE_APPLIED
    if _SAFE_APPLIED:
        return

    from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.nn import (
        attention as _attn,
    )
    from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.nn import (
        flexi_patch_embed as _fpe,
    )
    from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.nn import (
        flexi_vit as _fv,
    )

    _attn.logger = _NoLogger()
    _fv.logger = _NoLogger()
    _fpe.logger = _NoLogger()

    def _list_intersection_preserve_order(
        available_modalities, supported_modality_names
    ):
        supported = list(supported_modality_names)
        return [m for m in available_modalities if m in supported]

    _fv.get_modalities_to_process = _list_intersection_preserve_order

    _SAFE_APPLIED = True


def apply_s2_specialized_patches() -> None:
    """Replace upstream forward methods with S2-L2A-only specialized
    versions. Together with ``apply_safe_patches()`` this makes
    ``OlmoEarthWrapper(..."base").encoder`` fullgraph-compatible.

    These patches are NOT safe for general use:
      * ``MultiModalPatchEmbeddings.forward`` is replaced with a version
        that always processes ``sentinel2_l2a`` and assumes its data is
        present. Other modalities in the same encoder will be ignored.
      * ``Encoder.apply_attn`` is replaced with a version that hardcodes
        ``fast_pass=True`` semantics: skip mask construction, skip
        register tokens, no token-exit, no flash-attn packing.

    Apply in the same process where you build the inference wrapper.
    Reset between training and inference is not supported.
    """
    apply_safe_patches()

    global _S2_APPLIED
    if _S2_APPLIED:
        return

    from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.nn import flexi_vit as _fv

    # ----- MultiModalPatchEmbeddings: hardcode S2 -----
    def _patched_pe_forward(self, input_data, patch_size):
        modality = "sentinel2_l2a"
        modality_tokens, modality_masks = self.apply_embedding_to_modality(
            modality, input_data, patch_size
        )
        return {
            modality: modality_tokens,
            f"{modality}_mask": modality_masks,
        }

    _fv.MultiModalPatchEmbeddings.forward = _patched_pe_forward

    # ----- Encoder.apply_attn: hardcode fast_pass + S2-only -----
    def _patched_apply_attn(
        self,
        x,
        timestamps,
        patch_size,
        input_res,
        token_exit_cfg=None,
        fast_pass: bool = True,
    ):
        modality = "sentinel2_l2a"
        s2_tokens = x[modality]
        s2_mask = x[f"{modality}_mask"]

        # Composite encodings — call the per-modality method directly,
        # avoiding the dispatch dict / list construction.
        s2_encoded = self.composite_encodings._apply_encodings_per_modality(
            modality,
            s2_tokens,
            timestamps=timestamps,
            patch_size=patch_size,
            input_res=input_res,
        )

        # Flatten (b, h, w, t, b_s, d) -> (b, h*w*t*b_s, d) directly.
        b, ph, pw, t, bs, d = s2_encoded.shape
        tokens = s2_encoded.reshape(b, ph * pw * t * bs, d)

        # No register tokens for our base config; no flash-attn; no mask.
        for blk in self.blocks:
            tokens = blk(x=tokens, attn_mask=None)

        tokens = self.norm(tokens)

        # Reshape back into per-modality dict format the caller expects.
        tokens_per_modality_dict = {
            modality: tokens.reshape(b, ph, pw, t, bs, d),
            f"{modality}_mask": s2_mask,
        }
        return tokens_per_modality_dict, None

    _fv.Encoder.apply_attn = _patched_apply_attn

    _S2_APPLIED = True


def reset_for_test() -> None:
    """Test hook: undo the in-memory flag (does NOT restore patched
    methods). Re-import ``olmoearth_pretrain_minimal`` modules in a fresh
    interpreter to truly revert.
    """
    global _SAFE_APPLIED, _S2_APPLIED
    _SAFE_APPLIED = False
    _S2_APPLIED = False
