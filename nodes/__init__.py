"""ComfyUI-UniDepth nodes registration."""

import os
import sys

# Vendored upstream package uses absolute imports like `from unidepth.models...`.
# Put the directory containing the vendored `unidepth/` on sys.path so those work.
_NODES_DIR = os.path.dirname(os.path.abspath(__file__))
if _NODES_DIR not in sys.path:
    sys.path.insert(0, _NODES_DIR)

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
