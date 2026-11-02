"""Level 6: convolution and pooling. Images are (N, C, H, W): batch, channels, height, width."""
import numpy as np
from .tensor import Tensor
from .nn import Module


def im2col(x, k, stride, padding):
    """Unroll every k x k patch of x into one row, so convolution becomes ONE matmul.

    x: array (N, C, H, W). Returns (cols, Ho, Wo) where cols has shape (N*Ho*Wo, C*k*k).
    Ho = (H + 2*padding - k) // stride + 1, and the same for Wo.

    TODO(kavya), Level 6.
      1. Pad H and W with np.pad.
      2. Make cols = np.empty((N, C, k, k, Ho, Wo)).
      3. For each kernel offset (i, j): cols[:, :, i, j] = the padded x sliced
         from i with step `stride`, Ho values long (same for j / Wo).
         Only k*k loop iterations, not one per pixel.
      4. Transpose to (N, Ho, Wo, C, k, k) and reshape to (N*Ho*Wo, C*k*k).
    """
    raise NotImplementedError("TODO(kavya): im2col")


def col2im(cols, x_shape, k, stride, padding):
    """The reverse of im2col, used in backward. Overlapping patches ADD (+=), not overwrite.

    TODO(kavya), Level 6. Undo step 4 of im2col, then loop over (i, j) and `+=` each
    slice into a zero padded array. Finally cut the padding off.
    """
    raise NotImplementedError("TODO(kavya): col2im")


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, seed=None):
        rng = np.random.default_rng(seed)
        fan_in = in_channels * kernel_size * kernel_size
        self.W = Tensor(rng.normal(0, np.sqrt(2.0 / fan_in), (out_channels, in_channels, kernel_size, kernel_size)))
        self.b = Tensor(np.zeros(out_channels))
        self.k, self.stride, self.padding = kernel_size, stride, padding

    def forward(self, x):
        # TODO(kavya), Level 6:
        #   cols, Ho, Wo = im2col(...)                  (N*Ho*Wo, C*k*k)
        #   Wm = self.W.data.reshape(F, -1)             (F, C*k*k)
        #   out = cols @ Wm.T + b                       (N*Ho*Wo, F)
        #   -> reshape to (N, Ho, Wo, F) -> transpose to (N, F, Ho, Wo)
        # Backward: undo that transpose/reshape on out.grad to get g (N*Ho*Wo, F), then
        #   b.grad += g summed over rows
        #   W.grad += (g.T @ cols) reshaped to W's shape
        #   x.grad += col2im(g @ Wm, ...)
        # These are exactly the Linear-layer gradients from Level 3.
        raise NotImplementedError("TODO(kavya): Conv2d.forward")


class MaxPool2d(Module):
    def __init__(self, kernel_size=2):
        self.k = kernel_size

    def forward(self, x):
        # TODO(kavya), Level 6. For k=2 and H, W divisible by k:
        #   r = x.data.reshape(N, C, H//k, k, W//k, k);  out = r.max(axis=(3, 5))
        # Backward: only the max in each window gets gradient. Build a mask
        # (r == out expanded back), divide by mask.sum over the window (for ties),
        # multiply by out.grad expanded the same way, reshape to x's shape.
        raise NotImplementedError("TODO(kavya): MaxPool2d.forward")


class Flatten(Module):
    def forward(self, x):
        # TODO(kavya): (N, C, H, W) -> (N, C*H*W). One line.
        raise NotImplementedError("TODO(kavya): Flatten.forward")
