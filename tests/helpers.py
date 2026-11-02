import numpy as np
import torch


def t(x):
    """NumPy array -> float64 torch tensor that tracks gradients."""
    return torch.tensor(np.asarray(x, dtype=np.float64), requires_grad=True)


def close(a, b, tol=1e-6):
    b = b.detach().numpy() if isinstance(b, torch.Tensor) else b
    return np.allclose(a, b, atol=tol, rtol=tol)
