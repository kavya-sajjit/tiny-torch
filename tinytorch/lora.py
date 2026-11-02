"""Level 7 (bonus): LoRA, Hu et al. 2021.

Instead of fine-tuning a big weight W (in x out), keep W frozen and learn a
low-rank update:  y = x @ W + b + (x @ A @ B) * (alpha / r)
with A: (in, r), B: (r, out), and r much smaller than in and out.
This is the same layer you'll write in PyTorch for the LoRA paper project.
"""
import numpy as np
from .tensor import Tensor
from .nn import Module, Linear


class LoRALinear(Module):
    def __init__(self, base, r=4, alpha=8, seed=None):
        rng = np.random.default_rng(seed)
        self.base = base
        # TODO(kavya):
        #   1. Freeze the base layer: set requires_grad = False on base.W and base.b
        #      (Module.parameters() skips frozen tensors).
        #   2. self.A = Tensor of small random normals (std 0.01), shape (in, r)
        #   3. self.B = Tensor of ZEROS, shape (r, out). Why zeros? Read section 4.1.
        #   4. self.scale = alpha / r
        raise NotImplementedError("TODO(kavya): LoRALinear.__init__")

    def forward(self, x):
        # TODO(kavya): base output + the low-rank path, scaled.
        raise NotImplementedError("TODO(kavya): LoRALinear.forward")

    def merge(self):
        """Return a plain Linear whose W already includes the LoRA update,
        so inference costs nothing extra."""
        # TODO(kavya)
        raise NotImplementedError("TODO(kavya): LoRALinear.merge")
