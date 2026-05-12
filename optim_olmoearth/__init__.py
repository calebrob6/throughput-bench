"""Optimized OlmoEarth-Base inference variants.

All variants must be drop-in equivalent to ``geo_models.OlmoEarthWrapper``:
take a ``(B, 12, 128, 128)`` float tensor on a CUDA device and return a
``(B, 768)`` float tensor matching the reference output to the precision
allowed by the tolerance table in ``optim_olmoearth.equiv``.
"""
