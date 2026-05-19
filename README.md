> [!WARNING]
> Warning, uses experimental package `comfy-env` to attempt a one click isolated install. Will download and use pixi package manager.

# ComfyUI-UniDepth

ComfyUI nodes for [**UniDepthV2**](https://github.com/lpiccinelli-eth/UniDepth) — universal **monocular metric depth** estimation by Piccinelli et al. (ETH Zürich, CVPR 2024 + 2025 follow-up).

From a single RGB image, UniDepth predicts:

- Metric depth (meters)
- Camera intrinsics (or honours yours if you provide them)
- Per-pixel 3D point map (XYZ in camera space)
- Per-pixel ray directions
- Confidence map

## Nodes

| Node | Purpose |
|---|---|
| `UniDepthLoader` | Load a UniDepth model (V2 ViT-S/B/L or V1 ViT-L / ConvNeXt-L). Auto-downloads from HuggingFace into `ComfyUI/models/unidepth/`. |
| `UniDepthInfer` | Run inference. RGB image in → metric depth, points, intrinsics, rays, confidence. |
| `UniDepthCameraIntrinsics` | Build a Pinhole 3×3 K from fx/fy/cx/cy (or fov + image size). |
| `UniDepthDepthPreview` | Colorize raw metric depth to an `IMAGE` for visual inspection. |
| `UniDepthSavePointCloud` | Save the per-pixel point map as a colored PLY. |

## License

- **Wrapper code** (everything outside `nodes/unidepth/`): MIT — see `LICENSE`.
- **Vendored upstream source** (`nodes/unidepth/`) and **pretrained weights** downloaded at runtime: **CC-BY-NC 4.0** — see `LICENSE-UPSTREAM` and `NOTICE`.

Because the model weights and the vendored upstream code are non-commercial, **end-user use of this node pack is non-commercial only.**

## Citation

```bibtex
@inproceedings{piccinelli2024unidepth,
  title     = {UniDepth: Universal Monocular Metric Depth Estimation},
  author    = {Piccinelli, Luigi and Yang, Yung-Hsu and Sakaridis, Christos and Segu, Mattia and Li, Siyuan and Van Gool, Luc and Yu, Fisher},
  booktitle = {CVPR},
  year      = {2024}
}

@misc{piccinelli2025unidepthv2,
  title = {UniDepthV2: Universal Monocular Metric Depth Estimation Made Simpler},
  author = {Piccinelli, Luigi and Sakaridis, Christos and Yang, Yung-Hsu and Segu, Mattia and Li, Siyuan and Abbeloos, Wim and Van Gool, Luc},
  year = {2025},
  eprint = {2502.20110},
  archivePrefix = {arXiv}
}
```
