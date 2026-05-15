"""Shared utilities for ComfyUI-UniDepth nodes."""

import logging

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger("ComfyUI-UniDepth")

PATCH_SIZE = 14
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

MODEL_REPOS = {
    "unidepth-v2-vits14": "lpiccinelli/unidepth-v2-vits14",
    "unidepth-v2-vitb14": "lpiccinelli/unidepth-v2-vitb14",
    "unidepth-v2-vitl14": "lpiccinelli/unidepth-v2-vitl14",
    "unidepth-v1-vitl14": "lpiccinelli/unidepth-v1-vitl14",
    "unidepth-v1-cnvnxtl": "lpiccinelli/unidepth-v1-cnvnxtl",
}


def comfy_image_to_uint8_chw(images_bhwc: torch.Tensor) -> torch.Tensor:
    """ComfyUI IMAGE (B, H, W, C in [0, 1]) -> (B, C, H, W) float in [0, 255].

    UniDepth's `model.infer` normalizes internally as `rgb.float() / 255.0`, so we
    must hand it the 0..255 range; do NOT pre-normalize.
    """
    if images_bhwc.dim() != 4:
        raise ValueError(f"Expected IMAGE with shape (B,H,W,C), got {tuple(images_bhwc.shape)}")
    x = images_bhwc.clamp(0.0, 1.0).permute(0, 3, 1, 2).contiguous()
    return (x * 255.0).to(torch.float32)


def chw_to_comfy_image(chw: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> (B, H, W, C). If C=1, replicate to 3."""
    if chw.dim() == 3:
        chw = chw.unsqueeze(0)
    if chw.shape[1] == 1:
        chw = chw.repeat(1, 3, 1, 1)
    return chw.permute(0, 2, 3, 1).contiguous().cpu().float()


def normalize_for_preview(t: torch.Tensor, vmin=None, vmax=None) -> torch.Tensor:
    """Min-max normalize a tensor to [0, 1] for preview. Operates per-batch element."""
    if t.numel() == 0:
        return t
    if t.dim() == 3:  # (C,H,W)
        t = t.unsqueeze(0)
    out = torch.empty_like(t, dtype=torch.float32)
    for i in range(t.shape[0]):
        x = t[i].float()
        lo = float(x.min()) if vmin is None else float(vmin)
        hi = float(x.max()) if vmax is None else float(vmax)
        if hi - lo < 1e-8:
            out[i] = torch.zeros_like(x)
        else:
            out[i] = (x - lo) / (hi - lo)
    return out.clamp(0.0, 1.0)


# A handful of perceptually-uniform colormaps, ported to numpy so we don't pull
# matplotlib into the inference env.
_COLORMAPS = {
    "magma_r": None,  # filled lazily
    "magma": None,
    "viridis": None,
    "turbo": None,
    "inferno": None,
    "gray": None,
}


def _build_colormaps():
    """Build LUTs lazily. Uses matplotlib if available, else falls back to a
    simple turbo-style ramp computed in numpy."""
    try:
        import matplotlib.cm as cm  # noqa: PLC0415

        for name in list(_COLORMAPS.keys()):
            cmap = cm.get_cmap(name)
            lut = (np.asarray([cmap(i / 255.0)[:3] for i in range(256)]) * 255).astype(np.uint8)
            _COLORMAPS[name] = lut
    except Exception:
        # Fallback: build a few hand-rolled ramps.
        idx = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        gray = np.stack([idx, idx, idx], axis=-1)
        turbo = np.stack([
            np.clip(1.5 - 4.0 * np.abs(idx - 0.75), 0, 1),
            np.clip(1.5 - 4.0 * np.abs(idx - 0.50), 0, 1),
            np.clip(1.5 - 4.0 * np.abs(idx - 0.25), 0, 1),
        ], axis=-1)
        magma = np.stack([np.clip(idx ** 0.5 * 1.2, 0, 1),
                          np.clip(idx ** 1.5, 0, 1),
                          np.clip(0.6 * (1 - idx) + 0.4 * idx, 0, 1)], axis=-1)
        for k, v in {"gray": gray, "turbo": turbo, "viridis": magma,
                     "inferno": magma, "magma": magma,
                     "magma_r": magma[::-1]}.items():
            _COLORMAPS[k] = (v * 255).astype(np.uint8)


def colorize_depth(
    depth: torch.Tensor,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "magma_r",
    invert: bool = False,
) -> torch.Tensor:
    """Colorize a depth/scalar field to an RGB IMAGE tensor.

    Args:
        depth: float tensor, shape (B,1,H,W) | (B,H,W) | (H,W).
        vmin/vmax: range; None -> per-batch min/max.
        cmap: one of `_COLORMAPS`.
        invert: invert the colormap.

    Returns:
        (B, H, W, 3) float in [0, 1].
    """
    if _COLORMAPS["magma_r"] is None:
        _build_colormaps()
    lut = _COLORMAPS.get(cmap)
    if lut is None:
        lut = _COLORMAPS["magma_r"]
    if invert:
        lut = lut[::-1]

    if depth.dim() == 2:
        depth = depth.unsqueeze(0).unsqueeze(0)
    elif depth.dim() == 3:
        depth = depth.unsqueeze(1)

    out = []
    for i in range(depth.shape[0]):
        d = depth[i, 0].detach().float().cpu().numpy()
        lo = float(np.nanmin(d)) if vmin is None else float(vmin)
        hi = float(np.nanmax(d)) if vmax is None else float(vmax)
        if hi - lo < 1e-8:
            idx = np.zeros_like(d, dtype=np.uint8)
        else:
            idx = np.clip(((d - lo) / (hi - lo)) * 255.0, 0, 255).astype(np.uint8)
        rgb = lut[idx].astype(np.float32) / 255.0  # (H, W, 3)
        out.append(torch.from_numpy(rgb))
    return torch.stack(out, dim=0)


def resize_to_patch_multiple(images_pt: torch.Tensor, patch_size: int = PATCH_SIZE):
    """Resize input to nearest patch_size multiple via bilinear interpolation.

    UniDepth's `infer` does its own bucketing, but pre-aligning avoids the
    inner pad step and makes output sizes predictable.
    """
    _, _, H, W = images_pt.shape
    new_H = max(patch_size, ((H + patch_size // 2) // patch_size) * patch_size)
    new_W = max(patch_size, ((W + patch_size // 2) // patch_size) * patch_size)
    if (new_H, new_W) == (H, W):
        return images_pt, (H, W)
    return (
        F.interpolate(images_pt, size=(new_H, new_W), mode="bilinear", align_corners=False),
        (H, W),
    )


def save_points_to_ply(
    points: torch.Tensor,
    colors: torch.Tensor | None,
    out_path: str,
    valid_mask: torch.Tensor | None = None,
) -> str:
    """Write a colored point cloud (per-pixel) to a binary PLY file.

    Args:
        points: (3, H, W) or (H, W, 3) XYZ float tensor.
        colors: (H, W, 3) or (3, H, W) float in [0, 1], or None.
        out_path: target path; parent dirs are created.
        valid_mask: optional bool (H, W) - keep only True pixels.

    Returns the written path.
    """
    import os

    if points.dim() == 4:
        points = points[0]
    if points.shape[0] == 3 and points.dim() == 3:
        points = points.permute(1, 2, 0)  # (H, W, 3)
    H, W, _ = points.shape

    if colors is None:
        colors = torch.full((H, W, 3), 0.5)
    if colors.dim() == 4:
        colors = colors[0]
    if colors.shape[0] == 3 and colors.dim() == 3:
        colors = colors.permute(1, 2, 0)

    if valid_mask is None:
        valid_mask = torch.isfinite(points).all(dim=-1)

    pts = points[valid_mask].detach().cpu().float().numpy()
    cols = (colors[valid_mask].detach().cpu().float().clamp(0, 1).numpy() * 255).astype(np.uint8)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    n = pts.shape[0]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")

    dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("r", "u1"), ("g", "u1"), ("b", "u1"),
    ])
    arr = np.empty(n, dtype=dtype)
    arr["x"] = pts[:, 0]
    arr["y"] = pts[:, 1]
    arr["z"] = pts[:, 2]
    arr["r"] = cols[:, 0]
    arr["g"] = cols[:, 1]
    arr["b"] = cols[:, 2]

    with open(out_path, "wb") as f:
        f.write(header)
        arr.tofile(f)

    return out_path
