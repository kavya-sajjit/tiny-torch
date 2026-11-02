"""Level 1: scalar autograd. Each test compares your Value against PyTorch."""
import math
import pytest
import torch

from tinytorch.value import Value


def test_add_and_mul_forward():
    a, b = Value(2.0), Value(-3.0)
    assert (a + b).data == -1.0
    assert (a * b).data == -6.0
    assert (a + 1).data == 3.0 and (2 * a).data == 4.0


def test_add_mul_backward():
    a, b = Value(2.0), Value(-3.0)
    c = a * b + a          # dc/da = b + 1 = -2, dc/db = a = 2
    c.backward()
    assert a.grad == pytest.approx(-2.0)
    assert b.grad == pytest.approx(2.0)


def test_same_value_used_twice_accumulates_grad():
    a = Value(3.0)
    b = a + a              # db/da = 2, not 1
    b.backward()
    assert a.grad == pytest.approx(2.0)


def test_pow_relu_tanh_exp():
    for fn_v, fn_t in [
        (lambda x: x ** 3, lambda x: x ** 3),
        (lambda x: x.relu(), torch.relu),
        (lambda x: x.tanh(), torch.tanh),
        (lambda x: x.exp(), torch.exp),
    ]:
        for x0 in (-1.5, 0.7):
            v = Value(x0)
            y = fn_v(v)
            y.backward()
            t = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
            yt = fn_t(t)
            yt.backward()
            assert y.data == pytest.approx(yt.item())
            assert v.grad == pytest.approx(t.grad.item())


def test_matches_pytorch_on_a_bigger_expression():
    a, b = Value(-4.0), Value(2.0)
    c = a + b
    d = a * b + b ** 3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e ** 2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()

    at = torch.tensor(-4.0, dtype=torch.float64, requires_grad=True)
    bt = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    c = at + bt
    d = at * bt + bt ** 3
    c = c + c + 1
    c = c + 1 + c + (-at)
    d = d + d * 2 + (bt + at).relu()
    d = d + 3 * d + (bt - at).relu()
    e = c - d
    f = e ** 2
    gt = f / 2.0
    gt = gt + 10.0 / f
    gt.backward()

    assert g.data == pytest.approx(gt.item())
    assert a.grad == pytest.approx(at.grad.item())
    assert b.grad == pytest.approx(bt.grad.item())


def test_a_tiny_neuron():
    # out = tanh(w1*x1 + w2*x2 + b): the classic micrograd example
    x1, x2, w1, w2 = Value(2.0), Value(0.0), Value(-3.0), Value(1.0)
    b = Value(6.8813735870195432)
    out = (x1 * w1 + x2 * w2 + b).tanh()
    out.backward()
    assert out.data == pytest.approx(math.sqrt(0.5), abs=1e-6)
    assert w1.grad == pytest.approx(1.0, abs=1e-6)
    assert x1.grad == pytest.approx(-1.5, abs=1e-6)
    assert w2.grad == pytest.approx(0.0, abs=1e-6)
