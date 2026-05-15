"""Inference-only loss stubs.

The upstream `unidepth.ops.losses` module is the training-time loss zoo and
imports a custom CUDA `extract_patches` extension. We do not need it for
inference, but `UniDepthV{1,2}.__init__` always calls `build_losses(config)`
which does:

    mod = importlib.import_module("unidepth.ops.losses")
    loss_factory = getattr(mod, loss_config["name"])
    self.losses[loss_name] = loss_factory.build(loss_config)

So we just provide every loss class name as a no-op factory whose `.build()`
classmethod returns a tiny module that swallows any call and returns zero.
"""

import torch
import torch.nn as nn


class _NoopLoss(nn.Module):
    """A module that accepts any args/kwargs and returns a zero scalar."""

    name: str = "noop"
    weight: float = 0.0

    def __init__(self, name: str = "noop", weight: float = 0.0):
        super().__init__()
        self.name = name
        self.weight = float(weight)

    def forward(self, *args, **kwargs):
        return torch.zeros((), dtype=torch.float32)

    __call__ = forward


class _LossFactory:
    """Class-with-`.build()` shim matching `loss_factory.build(loss_config)`."""

    name = "noop"

    @classmethod
    def build(cls, loss_config):
        return _NoopLoss(name=loss_config.get("name", cls.name),
                         weight=loss_config.get("weight", 0.0))


# Loss class names referenced anywhere in the upstream configs.
class ARel(_LossFactory): name = "ARel"
class Confidence(_LossFactory): name = "Confidence"
class Dummy(_LossFactory): name = "Dummy"
class EdgeGuidedLocalSSI(_LossFactory): name = "EdgeGuidedLocalSSI"
class LocalSSI(_LossFactory): name = "LocalSSI"
class Regression(_LossFactory): name = "Regression"
class SelfDistill(_LossFactory): name = "SelfDistill"
class SILog(_LossFactory): name = "SILog"
class TeacherDistill(_LossFactory): name = "TeacherDistill"


__all__ = [
    "ARel", "Confidence", "Dummy", "EdgeGuidedLocalSSI", "LocalSSI",
    "Regression", "SelfDistill", "SILog", "TeacherDistill",
]
