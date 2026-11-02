"""Level 4: Module, Linear, ReLU, Sequential, MSE and SGD. An MLP learns XOR."""
import numpy as np
import torch

from tinytorch.tensor import Tensor
from tinytorch.nn import Linear, ReLU, Sequential, mse_loss
from tinytorch.optim import SGD
from helpers import t, close


def test_linear_matches_pytorch():
    layer = Linear(3, 2, seed=0)
    x0 = np.random.default_rng(0).normal(size=(4, 3))
    out = layer(Tensor(x0))
    out.sum().backward()
    W, b, xt = t(layer.W.data), t(layer.b.data), t(x0)
    (xt @ W + b).sum().backward()
    assert out.shape == (4, 2)
    assert close(layer.W.grad, W.grad) and close(layer.b.grad, b.grad)


def test_parameters_are_collected():
    model = Sequential(Linear(2, 8), ReLU(), Linear(8, 1))
    assert len(model.parameters()) == 4


def test_mse_loss():
    p, y = Tensor(np.array([1.0, 2.0, 3.0])), np.array([1.0, 0.0, 0.0])
    loss = mse_loss(p, y)
    loss.backward()
    assert np.isclose(loss.data, (0 + 4 + 9) / 3)
    assert np.allclose(p.grad, 2 * (p.data - y) / 3)


def test_sgd_step_and_momentum():
    w = Tensor(np.array([1.0]))
    w.grad = np.array([0.5])
    opt = SGD([w], lr=0.1)
    opt.step()
    assert np.allclose(w.data, 0.95)

    w = Tensor(np.array([1.0]))
    opt = SGD([w], lr=0.1, momentum=0.9)
    for _ in range(2):
        w.grad = np.array([1.0])
        opt.step()
    # velocity: 1, then 0.9*1 + 1 = 1.9  -> 1 - 0.1*1 - 0.1*1.9
    assert np.allclose(w.data, 1 - 0.1 - 0.19)


def test_mlp_learns_xor():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)
    model = Sequential(Linear(2, 16, seed=1), ReLU(), Linear(16, 1, seed=2))
    opt = SGD(model.parameters(), lr=0.1, momentum=0.9)
    for _ in range(500):
        opt.zero_grad()
        loss = mse_loss(model(Tensor(X)), y)
        loss.backward()
        opt.step()
    assert loss.data < 0.01
