"""Level 4: layers, built on top of Tensor."""
import numpy as np
from .tensor import Tensor


class Module:
    """Base class. Given: read it so you know how parameters() finds your weights."""

    def __call__(self, *args):
        return self.forward(*args)

    def forward(self, *args):
        raise NotImplementedError

    def parameters(self):
        params = []
        for v in vars(self).values():
            if isinstance(v, Tensor):
                if v.requires_grad:
                    params.append(v)
            elif isinstance(v, Module):
                params.extend(v.parameters())
            elif isinstance(v, (list, tuple)):
                for item in v:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
        return params

    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)


class Linear(Module):
    def __init__(self, in_features, out_features, seed=None):
        rng = np.random.default_rng(seed)
        # Given: He initialisation, good for ReLU networks. W is (in, out), so y = x @ W + b.
        self.W = Tensor(rng.normal(0, np.sqrt(2.0 / in_features), (in_features, out_features)))
        self.b = Tensor(np.zeros(out_features))

    def forward(self, x):
        # TODO(kavya): one line.
        raise NotImplementedError("TODO(kavya): Linear.forward")


class ReLU(Module):
    def forward(self, x):
        # TODO(kavya): one line.
        raise NotImplementedError("TODO(kavya): ReLU.forward")


class Sequential(Module):
    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x):
        # TODO(kavya): pass x through each layer in order.
        raise NotImplementedError("TODO(kavya): Sequential.forward")


def mse_loss(pred, target):
    # TODO(kavya): mean of squared differences, using only Tensor ops
    # (so backward works for free).
    raise NotImplementedError("TODO(kavya): mse_loss")
