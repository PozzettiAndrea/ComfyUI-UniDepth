"""Inference node for UniDepth."""

import torch
from comfy.utils import ProgressBar
from comfy_api.latest import io

from .utils import (
    chw_to_comfy_image,
    colorize_depth,
    comfy_image_to_uint8_chw,
    logger,
)


def _mm():
    import comfy.model_management

    return comfy.model_management


class UniDepthInfer(io.ComfyNode):
    """Run UniDepth on RGB images. Loops per-frame (mixed aspect ratios are safe)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UniDepthInfer",
            display_name="UniDepth Inference",
            category="UniDepth",
            description=(
                "Predict metric depth, intrinsics, point cloud, rays and "
                "confidence from a single RGB image (per batch element).\n\n"
                "If you pass a 3x3 INTRINSICS, the model uses it as a prior; "
                "otherwise it predicts intrinsics itself. The predicted K is "
                "always exposed on the `intrinsics_pred` output."
            ),
            inputs=[
                io.Custom("UNIDEPTH_MODEL").Input("unidepth_model"),
                io.Image.Input("images"),
                io.Custom("INTRINSICS").Input("intrinsics", optional=True),
                io.Float.Input(
                    "depth_vmin",
                    default=0.01,
                    min=0.0,
                    max=1000.0,
                    step=0.01,
                    optional=True,
                    tooltip="vmin for the colorized depth preview (meters).",
                ),
                io.Float.Input(
                    "depth_vmax",
                    default=10.0,
                    min=0.0,
                    max=1000.0,
                    step=0.1,
                    optional=True,
                    tooltip="vmax for the colorized depth preview (meters).",
                ),
                io.Combo.Input(
                    "preview_cmap",
                    options=["magma_r", "magma", "viridis", "turbo", "inferno", "gray"],
                    default="magma_r",
                    optional=True,
                ),
            ],
            outputs=[
                io.Image.Output(display_name="depth_preview"),
                io.Image.Output(display_name="depth_metric"),
                io.Image.Output(display_name="confidence"),
                io.Image.Output(display_name="rays"),
                io.Custom("INTRINSICS").Output(display_name="intrinsics_pred"),
                io.Custom("POINTS").Output(display_name="points"),
            ],
        )

    @classmethod
    def execute(
        cls,
        unidepth_model,
        images,
        intrinsics=None,
        depth_vmin=0.01,
        depth_vmax=10.0,
        preview_cmap="magma_r",
    ):
        from .load_model import _get_or_build_unidepth_model  # lazy

        model = _get_or_build_unidepth_model(unidepth_model)

        device = _mm().get_torch_device()
        model = model.to(device)

        B, H, W, C = images.shape
        logger.info(f"UniDepth inference on batch={B} HxW={H}x{W}")

        # ComfyUI IMAGE (BHWC, 0..1 float) -> (B,C,H,W) float 0..255 as the model expects.
        rgb_bchw = comfy_image_to_uint8_chw(images)

        # Camera prior (3x3 Pinhole K). Accept either a raw 3x3 tensor or
        # our INTRINSICS dict from UniDepthCameraIntrinsics.
        K = None
        if intrinsics is not None:
            if isinstance(intrinsics, dict):
                K = intrinsics.get("K")
            else:
                K = intrinsics
            if K is not None:
                K = torch.as_tensor(K, dtype=torch.float32)
                if K.dim() == 2:
                    K = K.unsqueeze(0)  # (1,3,3)
                K = K.to(device)

        pbar = ProgressBar(B)
        depth_metric_list = []
        confidence_list = []
        rays_list = []
        intrinsics_pred = []
        points_list = []

        for i in range(B):
            _mm().throw_exception_if_processing_interrupted()
            rgb_i = rgb_bchw[i].to(device)  # (C,H,W) 0..255

            cam_i = None
            if K is not None:
                cam_i = K[i] if K.shape[0] > 1 else K[0]

            with torch.no_grad():
                out = model.infer(rgb_i, cam_i)

            depth_metric_list.append(out["depth"].detach().float().cpu())
            confidence_list.append(out.get("confidence", torch.ones_like(out["depth"])).detach().float().cpu())
            rays_list.append(out["rays"].detach().float().cpu())
            intrinsics_pred.append(out["intrinsics"].detach().float().cpu())
            points_list.append(out["points"].detach().float().cpu())
            pbar.update(1)

        depth_metric = torch.cat(depth_metric_list, dim=0)  # (B,1,H,W)
        confidence = torch.cat(confidence_list, dim=0)
        rays = torch.cat(rays_list, dim=0)  # (B,3,H,W)
        intr = torch.cat(intrinsics_pred, dim=0)  # (B,3,3)
        points = torch.cat(points_list, dim=0)  # (B,3,H,W)

        depth_preview = colorize_depth(
            depth_metric, vmin=float(depth_vmin), vmax=float(depth_vmax), cmap=preview_cmap
        )  # (B,H,W,3)

        # Confidence is a scalar field whose range is not bounded to [0, 1]
        # (per-batch min/max varies). Normalize per-batch element for preview.
        conf_norm = torch.zeros_like(confidence)
        for i in range(confidence.shape[0]):
            c = confidence[i]
            lo, hi = float(c.min()), float(c.max())
            conf_norm[i] = (c - lo) / (hi - lo) if hi - lo > 1e-8 else torch.zeros_like(c)
        conf_image = chw_to_comfy_image(conf_norm)

        # Rays are unit vectors in [-1, 1] — shift to [0, 1] for display.
        rays_image = chw_to_comfy_image((rays.clamp(-1, 1) + 1.0) * 0.5)

        return io.NodeOutput(
            depth_preview,
            chw_to_comfy_image(depth_metric),
            conf_image,
            rays_image,
            {"K": intr},
            {"xyz": points},
        )
