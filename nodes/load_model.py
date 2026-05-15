"""Loader node for UniDepthV2 / UniDepthV1 models.

The actual model object is built lazily inside the inference node so this
loader stays JSON-safe across the comfy-env isolation boundary — same pattern
ComfyUI-DepthAnythingV3 uses.
"""

import logging
import os

import folder_paths
import torch
from comfy_api.latest import io

from .utils import MODEL_REPOS, logger

_unidepth_model_dir = os.path.join(folder_paths.models_dir, "unidepth")
os.makedirs(_unidepth_model_dir, exist_ok=True)
folder_paths.add_model_folder_path("unidepth", _unidepth_model_dir)


def _mm():
    import comfy.model_management

    return comfy.model_management


def _comfy_tqdm():
    """tqdm subclass that mirrors progress into ComfyUI's UI progress bar."""
    try:
        import comfy.utils
        import tqdm as _tqdm_mod
    except ImportError:
        return None

    holder = {"pbar": None, "total": 0, "done": 0}

    class _T(_tqdm_mod.tqdm):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            if self.total and self.total > 0 and holder["pbar"] is None:
                holder["total"] = self.total
                holder["done"] = 0
                holder["pbar"] = comfy.utils.ProgressBar(self.total)

        def update(self, n=1):
            ret = super().update(n)
            if n and holder["pbar"] and holder["total"] > 0:
                holder["done"] = min(holder["done"] + n, holder["total"])
                holder["pbar"].update_absolute(holder["done"], holder["total"])
            return ret

    return _T


_UNIDEPTH_MODEL_CACHE: dict[str, torch.nn.Module] = {}


def _get_or_build_unidepth_model(config: dict) -> torch.nn.Module:
    """Build (or fetch from cache) a UniDepth model from a JSON-safe config dict."""
    key = f"{config['model_name']}::{config['dtype']}"
    if key in _UNIDEPTH_MODEL_CACHE:
        return _UNIDEPTH_MODEL_CACHE[key]

    model_name = config["model_name"]
    repo_id = MODEL_REPOS[model_name]
    cache_dir = config.get("cache_dir") or _unidepth_model_dir

    # Import here to defer heavy CUDA / torch.hub init until first use.
    from unidepth.models import UniDepthV1, UniDepthV2  # vendored

    cls = UniDepthV1 if model_name.startswith("unidepth-v1-") else UniDepthV2

    logger.info(f"Loading {model_name} from {repo_id} (cache_dir={cache_dir})")
    model = cls.from_pretrained(repo_id, cache_dir=cache_dir)
    model = model.eval()

    # V2 exposes a resolution_level knob (0..9) and an interpolation mode.
    if hasattr(model, "interpolation_mode"):
        model.interpolation_mode = config.get("interpolation_mode", "bilinear")
    if config.get("resolution_level") is not None:
        model.resolution_level = int(config["resolution_level"])

    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map.get(config["dtype"], torch.float32)
    if dtype is not torch.float32:
        try:
            model = model.to(dtype)
        except Exception as e:
            logger.warning(f"Could not cast UniDepth to {dtype}: {e}; staying in fp32")

    _UNIDEPTH_MODEL_CACHE[key] = model
    return model


class UniDepthLoader(io.ComfyNode):
    """Download (if needed) and load a UniDepth model."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UniDepthLoader",
            display_name="UniDepth Loader",
            category="UniDepth",
            description=(
                "Load a UniDepth model from ComfyUI/models/unidepth/. "
                "First-time use will auto-download from HuggingFace "
                "(~600 MB ViT-S, ~1.0 GB ViT-B, ~1.3 GB ViT-L).\n\n"
                "License: CC-BY-NC 4.0 (non-commercial)."
            ),
            inputs=[
                io.Combo.Input(
                    "model_name",
                    options=list(MODEL_REPOS.keys()),
                    default="unidepth-v2-vitl14",
                    tooltip="Which UniDepth variant to load.",
                ),
                io.Combo.Input(
                    "dtype",
                    options=["fp32", "fp16", "bf16"],
                    default="fp32",
                    optional=True,
                    tooltip="Inference dtype. fp32 is the safest; fp16/bf16 cut VRAM.",
                ),
                io.Int.Input(
                    "resolution_level",
                    default=9,
                    min=0,
                    max=9,
                    optional=True,
                    tooltip=(
                        "V2 only: bucket index (0..9) over the model's pixel-count "
                        "range. Higher = sharper but slower. Ignored for V1."
                    ),
                ),
                io.Combo.Input(
                    "interpolation_mode",
                    options=["bilinear", "nearest-exact"],
                    default="bilinear",
                    optional=True,
                    tooltip="V2 only: postprocess interp mode for depth/points/rays.",
                ),
            ],
            outputs=[
                io.Custom("UNIDEPTH_MODEL").Output(display_name="unidepth_model"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model_name: str,
        dtype: str = "fp32",
        resolution_level: int = 9,
        interpolation_mode: str = "bilinear",
    ):
        # JSON-safe config; the actual model is built on first use by the
        # inference node via `_get_or_build_unidepth_model`.
        cfg = {
            "model_name": model_name,
            "dtype": dtype,
            "resolution_level": int(resolution_level),
            "interpolation_mode": interpolation_mode,
            "cache_dir": _unidepth_model_dir,
        }
        # Pre-warm the cache so the download progress shows under THIS node.
        try:
            _get_or_build_unidepth_model(cfg)
        except Exception as e:
            logging.exception("UniDepth model build failed")
            raise
        return io.NodeOutput(cfg)
