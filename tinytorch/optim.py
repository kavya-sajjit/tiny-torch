"""Levels 4 and 5: optimisers. They update p.data using p.grad."""
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = list(params)

    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.data)


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0):
        super().__init__(params)
        self.lr, self.momentum = lr, momentum
        self.v = [np.zeros_like(p.data) for p in self.params]   # one velocity per parameter

    def step(self):
        # TODO(kavya), Level 4. For each parameter p and its velocity v:
        #   v = momentum * v + grad
        #   p = p - lr * v
        # Careful: update v IN PLACE (v *= ..., v += ...), or self.v won't change.
        raise NotImplementedError("TODO(kavya): SGD.step")


class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params)
        self.lr, (self.b1, self.b2), self.eps = lr, betas, eps
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def step(self):
        # TODO(kavya), Level 5. Kingma & Ba 2014, Algorithm 1. Each step:
        #   t += 1
        #   m = b1*m + (1-b1)*g          (running mean of gradients)
        #   v = b2*v + (1-b2)*g**2       (running mean of squared gradients)
        #   m_hat = m / (1 - b1**t),  v_hat = v / (1 - b2**t)   (bias correction)
        #   p -= lr * m_hat / (sqrt(v_hat) + eps)
        # The test compares you to torch.optim.Adam step by step.
        raise NotImplementedError("TODO(kavya): Adam.step")
