"""Camera intrinsics builder node."""

import math

import torch
from comfy_api.latest import io


class UniDepthCameraIntrinsics(io.ComfyNode):
    """Build a Pinhole 3x3 intrinsics matrix from fx/fy/cx/cy or fov + size."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UniDepthCameraIntrinsics",
            display_name="UniDepth Camera Intrinsics (Pinhole)",
            category="UniDepth",
            description=(
                "Build a 3x3 Pinhole intrinsics matrix.\n\n"
                "If `fx` > 0 it is used directly (with `fy` = fx unless > 0). "
                "Otherwise fx and fy are derived from `fov_degrees` and the "
                "image size. `cx`/`cy` default to image center when 0."
            ),
            inputs=[
                io.Int.Input("image_width", default=1024, min=1, max=8192),
                io.Int.Input("image_height", default=768, min=1, max=8192),
                io.Float.Input("fx", default=0.0, min=0.0, max=100000.0, step=0.1, optional=True),
                io.Float.Input("fy", default=0.0, min=0.0, max=100000.0, step=0.1, optional=True),
                io.Float.Input("cx", default=0.0, min=0.0, max=100000.0, step=0.1, optional=True),
                io.Float.Input("cy", default=0.0, min=0.0, max=100000.0, step=0.1, optional=True),
                io.Float.Input(
                    "fov_degrees",
                    default=60.0,
                    min=1.0,
                    max=170.0,
                    step=0.5,
                    optional=True,
                    tooltip="Horizontal FOV in degrees, used only if `fx` == 0.",
                ),
            ],
            outputs=[
                io.Custom("INTRINSICS").Output(display_name="intrinsics"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image_width,
        image_height,
        fx=0.0,
        fy=0.0,
        cx=0.0,
        cy=0.0,
        fov_degrees=60.0,
    ):
        if fx <= 0.0:
            fx = (image_width * 0.5) / math.tan(math.radians(float(fov_degrees)) * 0.5)
        if fy <= 0.0:
            fy = fx
        if cx <= 0.0:
            cx = image_width * 0.5
        if cy <= 0.0:
            cy = image_height * 0.5

        K = torch.tensor(
            [
                [float(fx), 0.0, float(cx)],
                [0.0, float(fy), float(cy)],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        return io.NodeOutput({"K": K})
