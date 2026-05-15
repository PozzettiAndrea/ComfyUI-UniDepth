"""Install hook.

Uses the unresolved `__file__` parent so that when this pack is run via a
symlink inside ComfyUI/custom_nodes/, the comfy-env workspace detection walks
up the symlinked path (and finds ComfyUI's `main.py`) rather than walking up
the source location where this file actually lives.
"""

from pathlib import Path

from comfy_env import install

install(node_dir=Path(__file__).parent)
