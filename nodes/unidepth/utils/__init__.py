"""Slimmed-down package init for the ComfyUI inference path.

The upstream `__init__.py` re-exports symbols from `visualization` (which pulls
wandb / matplotlib) and `evaluation_depth` (which pulls a CUDA knn extension we
no longer ship). Neither is needed for inference, so we leave the package empty
and let consumers import the submodules they actually use directly.
"""
