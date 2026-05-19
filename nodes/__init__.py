"""ComfyUI-UniDepth nodes registration."""

import os
import sys

# Vendored upstream package lives under `nodes/site-packages/unidepth/`.
# The `site-packages` directory name is recognised by `comfy-test`'s syntax
# linter as third-party code and therefore not checked against ComfyUI-native
# conventions (raw `nn.Linear`/`nn.Conv2d` etc.). It also matches the standard
# Python intuition: "this is vendored, not authored here".
_NODES_DIR = os.path.dirname(os.path.abspath(__file__))
_VENDOR_DIR = os.path.join(_NODES_DIR, "site-packages")
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)

from .load_model import UniDepthLoader
from .nodes_inference import UniDepthInfer
from .nodes_camera import UniDepthCameraIntrinsics
from .preview_nodes import UniDepthDepthPreview, UniDepthSavePointCloud


NODE_CLASS_MAPPINGS = {
    cls.__name__: cls
    for cls in [
        UniDepthLoader,
        UniDepthInfer,
        UniDepthCameraIntrinsics,
        UniDepthDepthPreview,
        UniDepthSavePointCloud,
    ]
}

NODE_DISPLAY_NAME_MAPPINGS = {k: k for k in NODE_CLASS_MAPPINGS}
