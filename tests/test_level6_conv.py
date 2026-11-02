"""Level 6: Conv2d via im2col, MaxPool2d, Flatten."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tinytorch.tensor import Tensor
from tinytorch.conv import Conv2d, MaxPool2d, Flatten, im2col
from helpers import t, close

rng = np.random.default_rng(2)


def test_im2col_shape():
    x = rng.normal(size=(2, 3, 5, 5))
    cols, Ho, Wo = im2col(x, k=3, stride=1, padding=1)
    assert (Ho, Wo) == (5, 5)
    assert cols.shape == (2 * 5 * 5, 3 * 3 * 3)


@pytest.mark.parametrize("stride,padding", [(1, 0), (1, 1), (2, 1)])
def test_conv2d_matches_pytorch(stride, padding):
    x0 = rng.normal(size=(2, 3, 7, 7))
    conv = Conv2d(3, 4, 3, stride=stride, padding=padding, seed=0)
    conv.b.data = rng.normal(size=4)
    x = Tensor(x0)
    y = conv(x)
    (y * y).sum().backward()

    xt, Wt, bt = t(x0), t(conv.W.data), t(conv.b.data)
    yt = F.conv2d(xt, Wt, bt, stride=stride, padding=padding)
    (yt * yt).sum().backward()
    assert close(y.data, yt)
    assert close(x.grad, xt.grad) and close(conv.W.grad, Wt.grad) and close(conv.b.grad, bt.grad)


def test_maxpool_matches_pytorch():
    x0 = rng.normal(size=(2, 3, 6, 6))
    x = Tensor(x0)
    y = MaxPool2d(2)(x)
    (y * y).sum().backward()
    xt = t(x0)
    yt = F.max_pool2d(xt, 2)
    (yt * yt).sum().backward()
    assert close(y.data, yt) and close(x.grad, xt.grad)


def test_flatten():
    x = Tensor(rng.normal(size=(2, 3, 4, 4)))
    y = Flatten()(x)
    assert y.shape == (2, 48)
    y.sum().backward()
    assert x.grad.shape == (2, 3, 4, 4)
