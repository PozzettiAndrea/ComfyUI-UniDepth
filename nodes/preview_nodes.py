"""Preview / output nodes: depth colorization + PLY export."""

import os

import folder_paths
import torch
from comfy_api.latest import io

from .utils import colorize_depth, save_points_to_ply


class UniDepthDepthPreview(io.ComfyNode):
    """Colorize a raw metric depth IMAGE for visual inspection.

    The inference node already returns a colorized `depth_preview` output; use
    this node when you want to re-colorize the raw `depth_metric` IMAGE with a
    different vmin/vmax/colormap, or to colorize a depth coming from elsewhere.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UniDepthDepthPreview",
            display_name="UniDepth Depth Preview",
            category="UniDepth",
            description="Colorize a depth IMAGE with a configurable colormap and range.",
            inputs=[
                io.Image.Input(
                    "depth_image",
                    tooltip="A depth IMAGE tensor (B,H,W,3 with depth replicated, or single channel).",
                ),
                io.Float.Input("vmin", default=0.01, min=0.0, max=1000.0, step=0.01, optional=True),
                io.Float.Input("vmax", default=10.0, min=0.0, max=1000.0, step=0.1, optional=True),
                io.Combo.Input(
                    "cmap",
                    options=["magma_r", "magma", "viridis", "turbo", "inferno", "gray"],
                    default="magma_r",
                    optional=True,
                ),
                io.Boolean.Input("invert", default=False, optional=True),
            ],
            outputs=[
                io.Image.Output(display_name="preview"),
            ],
        )

    @classmethod
    def execute(cls, depth_image, vmin=0.01, vmax=10.0, cmap="magma_r", invert=False):
        # ComfyUI IMAGE is (B,H,W,3). Collapse to single channel (assume R==G==B
        # for depth images that came through `chw_to_comfy_image`).
        if depth_image.dim() == 4:
            depth = depth_image[..., 0]  # (B,H,W)
        else:
            depth = depth_image
        rgb = colorize_depth(depth, vmin=float(vmin), vmax=float(vmax), cmap=cmap, invert=invert)
        return io.NodeOutput(rgb)


class UniDepthSavePointCloud(io.ComfyNode):
    """Save the per-pixel point map from UniDepth as a colored PLY."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UniDepthSavePointCloud",
            display_name="UniDepth Save Point Cloud",
            category="UniDepth",
            is_output_node=True,
            description=(
                "Save the (3, H, W) point map from UniDepth as a binary PLY in "
                "ComfyUI/output/. RGB colors come from the source IMAGE; pixels "
                "with infinite/NaN coordinates are dropped."
            ),
            inputs=[
                io.Custom("POINTS").Input("points"),
                io.Image.Input("rgb_image"),
                io.String.Input(
                    "filename_prefix",
                    default="unidepth_points",
                    tooltip="File saved as <output_dir>/<prefix>_<index>.ply",
                ),
                io.Int.Input(
                    "batch_index",
                    default=0,
                    min=0,
                    max=1024,
                    optional=True,
                    tooltip="Which batch element to save.",
                ),
            ],
            outputs=[
                io.String.Output(display_name="ply_path"),
            ],
        )

    @classmethod
    def execute(cls, points, rgb_image, filename_prefix="unidepth_points", batch_index=0):
        xyz = points["xyz"] if isinstance(points, dict) else points
        # xyz: (B,3,H,W)
        if xyz.dim() == 3:
            xyz = xyz.unsqueeze(0)
        b = min(int(batch_index), xyz.shape[0] - 1)
        xyz_i = xyz[b]  # (3,H,W)

        rgb = rgb_image[b] if rgb_image.dim() == 4 else rgb_image  # (H,W,3)
        # Match resolution if needed.
        if rgb.shape[:2] != xyz_i.shape[1:]:
            rgb_chw = rgb.permute(2, 0, 1).unsqueeze(0)
            rgb_chw = torch.nn.functional.interpolate(
                rgb_chw, size=xyz_i.shape[1:], mode="bilinear", align_corners=False
            )
            rgb = rgb_chw.squeeze(0).permute(1, 2, 0)

        out_dir = folder_paths.get_output_directory()
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{filename_prefix}_{b:04d}.ply")

        path = save_points_to_ply(xyz_i, rgb, out_path)
        return io.NodeOutput(path)
