"""Level 1: scalar autograd.

A Value wraps one number. Every operation creates a new Value that remembers:
  - its inputs (`_prev`)
  - a small function (`_backward`) that pushes the output's gradient back to the inputs
Calling `.backward()` on the final Value runs every `_backward` in the right order.

The pattern you will repeat for every op:

    out = Value(<forward result>, (self, other), "<op name>")
    def _backward():
        self.grad  += <d out / d self>  * out.grad
        other.grad += <d out / d other> * out.grad
    out._backward = _backward
    return out

Why `+=` and not `=`? Because a Value can be used more than once (e.g. `a + a`),
and its gradient is the sum of the contributions from every place it is used.
"""
import math


class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        # TODO(kavya): forward is self.data + other.data.
        # Hint: d(a+b)/da = 1 and d(a+b)/db = 1.
        raise NotImplementedError("TODO(kavya): Value.__add__")

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        # TODO(kavya): forward is self.data * other.data.
        # Hint: d(a*b)/da = b.
        raise NotImplementedError("TODO(kavya): Value.__mul__")

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "only constant powers"
        # TODO(kavya): forward is self.data ** k.
        # Hint: the power rule.
        raise NotImplementedError("TODO(kavya): Value.__pow__")

    def relu(self):
        # TODO(kavya): forward is max(0, x).
        # Hint: the slope is 1 where the output is positive, else 0.
        raise NotImplementedError("TODO(kavya): Value.relu")

    def tanh(self):
        # TODO(kavya): use math.tanh.
        # Hint: d tanh(x)/dx = 1 - tanh(x)**2. Reuse the forward result.
        raise NotImplementedError("TODO(kavya): Value.tanh")

    def exp(self):
        # TODO(kavya): use math.exp.
        # Hint: e^x is its own derivative.
        raise NotImplementedError("TODO(kavya): Value.exp")

    def backward(self):
        # TODO(kavya):
        # 1. Build a list `topo` where every node comes AFTER all of its inputs
        #    (topological order). A recursive function with a `visited` set works.
        # 2. Set self.grad = 1.0 (d self / d self).
        # 3. Walk `topo` in REVERSE and call each node's _backward().
        raise NotImplementedError("TODO(kavya): Value.backward")

    # ---- given: these are built from the ops above, so they work once yours do
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __truediv__(self, other): return self * other ** -1
    def __rtruediv__(self, other): return other * self ** -1
