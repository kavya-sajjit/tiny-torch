"""Level 3: reductions, matmul, reshape, log and a stable cross-entropy."""
import numpy as np
import torch
import torch.nn.functional as F

from tinytorch.tensor import Tensor, cross_entropy
from helpers import t, close

rng = np.random.default_rng(1)


def test_sum_all_and_axis():
    x0 = rng.normal(size=(3, 4))
    for kwargs in [{}, {"axis": 0}, {"axis": 1}, {"axis": 1, "keepdims": True}]:
        x = Tensor(x0)
        y = (x.sum(**kwargs) * 2.0).sum()
        y.backward()
        xt = t(x0)
        dim = kwargs.get("axis")
        st = xt.sum() if dim is None else xt.sum(dim=dim, keepdim=kwargs.get("keepdims", False))
        (st * 2.0).sum().backward()
        assert close(y.data, (st * 2.0).sum())
        assert close(x.grad, xt.grad)


def test_mean():
    x0 = rng.normal(size=(6, 2))
    x = Tensor(x0)
    x.mean().backward()
    assert np.allclose(x.grad, 1 / 12)
    x = Tensor(x0)
    x.mean(axis=0).sum().backward()
    assert np.allclose(x.grad, 1 / 6)


def test_matmul_backward():
    a0, b0 = rng.normal(size=(5, 3)), rng.normal(size=(3, 4))
    a, b = Tensor(a0), Tensor(b0)
    y = (a @ b).sum()
    y.backward()
    at, bt = t(a0), t(b0)
    (at @ bt).sum().backward()
    assert close(a.grad, at.grad) and close(b.grad, bt.grad)


def test_reshape_and_log():
    x0 = rng.uniform(0.5, 2.0, size=(2, 6))
    x = Tensor(x0)
    y = (x.reshape(3, 4).log() * Tensor(np.arange(12.0).reshape(3, 4))).sum()
    y.backward()
    xt = t(x0)
    yt = (xt.reshape(3, 4).log() * torch.arange(12.0, dtype=torch.float64).reshape(3, 4)).sum()
    yt.backward()
    assert close(y.data, yt) and close(x.grad, xt.grad)


def test_cross_entropy_matches_pytorch():
    logits0 = rng.normal(size=(8, 10)) * 3
    targets = rng.integers(0, 10, size=8)
    logits = Tensor(logits0)
    loss = cross_entropy(logits, targets)
    loss.backward()
    lt = t(logits0)
    loss_t = F.cross_entropy(lt, torch.tensor(targets))
    loss_t.backward()
    assert close(loss.data, loss_t)
    assert close(logits.grad, lt.grad)


def test_cross_entropy_is_numerically_stable():
    logits = Tensor(np.array([[1000.0, 0.0, -1000.0]]))
    loss = cross_entropy(logits, [0])
    assert np.isfinite(loss.data) and loss.data < 1e-6
