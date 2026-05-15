"""Inference-only stub of unidepth.ops.

The upstream package re-exports loss classes from `unidepth.ops.losses`. Those
losses pull in a custom CUDA extension (`extract_patches`) that we strip for
the ComfyUI wrapper, so we expose a no-op `losses` submodule instead and skip
the scheduler (training-only).
"""

from . import losses  # noqa: F401
