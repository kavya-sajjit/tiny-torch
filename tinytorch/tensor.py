"""Levels 2 and 3: the same idea as Value, but on NumPy arrays.

Each op follows the Level 1 pattern. The one new problem is broadcasting:
if `a` has shape (4, 3) and `b` has shape (3,), then `a + b` has shape (4, 3),
but b.grad must still have shape (3,). `unbroadcast` fixes that.
"""
import numpy as np


def unbroadcast(grad, shape):
    """Sum `grad` down so it has `shape`.

    TODO(kavya), Level 2. Two steps:
      1. While grad has MORE dimensions than `shape`, sum over axis 0.
         ((2, 4, 3) -> (4, 3))
      2. For every axis where shape has size 1 but grad doesn't, sum over that
         axis with keepdims=True. ((4, 3) -> (4, 1) when shape is (4, 1))
    Why sum? An element that was copied to 4 places got 4 gradient contributions.
    """
    raise NotImplementedError("TODO(kavya): unbroadcast")


class Tensor:
    def __init__(self, data, _children=(), _op="", requires_grad=True):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self.requires_grad = requires_grad      # False = frozen (used by LoRA, Level 7)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, op={self._op!r})"

    @staticmethod
    def _wrap(x):
        """Turn plain numbers / arrays into constant Tensors."""
        return x if isinstance(x, Tensor) else Tensor(x, requires_grad=False)

    # ================================================================ Level 2
    def __add__(self, other):
        other = Tensor._wrap(other)
        # TODO(kavya): like Value.__add__, but pass each gradient through
        # unbroadcast(..., self.shape) / unbroadcast(..., other.shape).
        raise NotImplementedError("TODO(kavya): Tensor.__add__")

    def __mul__(self, other):
        other = Tensor._wrap(other)
        # TODO(kavya): elementwise product. Same as Value, plus unbroadcast.
        raise NotImplementedError("TODO(kavya): Tensor.__mul__")

    def __pow__(self, k):
        assert isinstance(k, (int, float))
        # TODO(kavya): elementwise power. No broadcasting involved.
        raise NotImplementedError("TODO(kavya): Tensor.__pow__")

    def relu(self):
        # TODO(kavya): np.maximum(x, 0). The mask (x > 0) is the slope.
        raise NotImplementedError("TODO(kavya): Tensor.relu")

    def exp(self):
        # TODO(kavya): np.exp.
        raise NotImplementedError("TODO(kavya): Tensor.exp")

    def backward(self):
        # TODO(kavya): same as Value.backward, with two changes:
        #   - start with self.grad = np.ones_like(self.data)
        #   - NumPy arrays can't go in a set, so track visited nodes by id(node)
        raise NotImplementedError("TODO(kavya): Tensor.backward")

    # ================================================================ Level 3
    def sum(self, axis=None, keepdims=False):
        # TODO(kavya): forward is self.data.sum(axis=axis, keepdims=keepdims).
        # Backward: every input element contributed with slope 1, so each one
        # receives the output gradient. If an axis was removed (keepdims=False),
        # put it back with np.expand_dims, then np.broadcast_to(self.shape).
        raise NotImplementedError("TODO(kavya): Tensor.sum")

    def mean(self, axis=None, keepdims=False):
        # TODO(kavya): one line, using self.sum(...) and multiplication.
        # How many elements are you averaging over when axis is None? When it isn't?
        raise NotImplementedError("TODO(kavya): Tensor.mean")

    def __matmul__(self, other):
        other = Tensor._wrap(other)
        # TODO(kavya): out = A @ B, with A (n, k) and B (k, m).
        # Hint: work out the shapes. dA must be (n, k) and dB must be (k, m),
        # and out.grad is (n, m). There's only one way to multiply them to get that.
        raise NotImplementedError("TODO(kavya): Tensor.__matmul__")

    def reshape(self, *shape):
        # TODO(kavya): the gradient just gets reshaped back.
        raise NotImplementedError("TODO(kavya): Tensor.reshape")

    def log(self):
        # TODO(kavya): d log(x)/dx = 1/x.
        raise NotImplementedError("TODO(kavya): Tensor.log")

    # ---- given
    def __neg__(self): return self * -1.0
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __sub__(self, other): return self + (-Tensor._wrap(other))
    def __rsub__(self, other): return Tensor._wrap(other) + (-self)
    def __truediv__(self, other):
        other = Tensor._wrap(other)
        return self * other ** -1.0


def cross_entropy(logits, targets):
    """Mean cross-entropy loss. logits: Tensor (N, C); targets: ints (N,).

    TODO(kavya), Level 3. Write it as ONE op with its own _backward:
      1. Stability: z = logits.data - logits.data.max(axis=1, keepdims=True)
         (subtracting a constant per row doesn't change softmax, but stops exp overflowing)
      2. log_softmax = z - log(sum(exp(z), axis=1, keepdims=True))
      3. loss = -mean of log_softmax[row i, targets[i]]
         (hint: log_softmax[np.arange(N), targets])
      4. Backward has a famous, simple form: (softmax - one_hot(targets)) / N,
         multiplied by out.grad.
    """
    raise NotImplementedError("TODO(kavya): cross_entropy")
