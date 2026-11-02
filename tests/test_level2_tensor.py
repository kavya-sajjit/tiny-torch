"""Level 2: elementwise ops on arrays, and gradients that respect broadcasting."""
import numpy as np
import pytest

from tinytorch.tensor import Tensor, unbroadcast
from helpers import t, close

rng = np.random.default_rng(0)


def test_unbroadcast_shapes():
    g = np.ones((4, 3))
    assert unbroadcast(g, (4, 3)).shape == (4, 3)
    assert unbroadcast(g, (3,)).shape == (3,)
    assert unbroadcast(g, (1, 3)).shape == (1, 3)
    assert unbroadcast(g, (4, 1)).shape == (4, 1)
    assert np.allclose(unbroadcast(g, (3,)), 4.0)       # summed over 4 rows
    assert np.allclose(unbroadcast(g, (4, 1)), 3.0)     # summed over 3 columns


@pytest.mark.parametrize("shape_a,shape_b", [((4, 3), (4, 3)), ((4, 3), (3,)), ((4, 1), (1, 3)), ((2, 4, 3), (4, 3))])
def test_add_mul_with_broadcasting(shape_a, shape_b):
    a0, b0 = rng.normal(size=shape_a), rng.normal(size=shape_b)
    a, b = Tensor(a0), Tensor(b0)
    y = a * b + a
    y.backward()
    at, bt = t(a0), t(b0)
    yt = at * bt + at
    yt.backward(torch_ones(yt))
    assert close(y.data, yt)
    assert a.grad.shape == shape_a and b.grad.shape == shape_b
    assert close(a.grad, at.grad) and close(b.grad, bt.grad)


def torch_ones(x):
    import torch
    return torch.ones_like(x)


def test_pow_relu_exp_sub_div():
    x0 = rng.normal(size=(5, 2))
    x = Tensor(x0)
    y = (x ** 2 - x.relu() + x.exp() / 2.0)
    y.backward()
    xt = t(x0)
    yt = (xt ** 2 - xt.relu() + xt.exp() / 2.0)
    yt.backward(torch_ones(yt))
    assert close(y.data, yt) and close(x.grad, xt.grad)


def test_tensor_used_twice():
    x = Tensor(np.array([1.0, 2.0]))
    y = x * x + x
    y.backward()
    assert np.allclose(x.grad, 2 * x.data + 1)
