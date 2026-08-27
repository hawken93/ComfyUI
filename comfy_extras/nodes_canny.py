from kornia.filters import canny
from typing_extensions import override

import comfy.model_management
from comfy_api.latest import ComfyExtension, io
import torch


class Canny(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Canny",
            display_name="Detect Edges (Canny)",
            search_aliases=["edge detection", "outline", "contour detection", "line art"],
            category="image/filters",
            essentials_category="Image Tools",
            inputs=[
                io.Image.Input("image"),
                io.Float.Input("low_threshold", default=0.4, min=0.01, max=0.99, step=0.01),
                io.Float.Input("high_threshold", default=0.8, min=0.01, max=0.99, step=0.01),
                io.Combo.Input("device", options=["gpu" if o == "default" else o for o in comfy.model_management.get_gpu_device_options()], optional=True, advanced=True, default="cpu"),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def detect_edge(cls, image, low_threshold, high_threshold):
        # Deprecated: use the V3 schema's `execute` method instead of this.
        return cls.execute(image, low_threshold, high_threshold)

    @classmethod
    def execute(cls, image, low_threshold, high_threshold, device="cpu") -> io.NodeOutput:
        device = comfy.model_management.pick_device_for_option("default" if device == "gpu" else device)
        output = canny(image.to(device=device, dtype=torch.float32).movedim(-1, 1), low_threshold, high_threshold)
        intr_device = comfy.model_management.intermediate_device(device)
        img_out = output[1].to(device=intr_device, dtype=comfy.model_management.intermediate_dtype()).repeat(1, 3, 1, 1).movedim(1, -1)
        return io.NodeOutput(img_out)


class CannyExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [Canny]


async def comfy_entrypoint() -> CannyExtension:
    return CannyExtension()
