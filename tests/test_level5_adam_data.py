"""Level 5: Adam and a DataLoader. Then run examples/train_mnist_mlp.py."""
import numpy as np
import torch

from tinytorch.tensor import Tensor
from tinytorch.optim import Adam
from tinytorch.data import DataLoader


def test_adam_matches_pytorch():
    w0 = np.array([1.0, -2.0, 3.0])
    w = Tensor(w0.copy())
    wt = torch.tensor(w0.copy(), requires_grad=True)
    opt = Adam([w], lr=0.1)
    opt_t = torch.optim.Adam([wt], lr=0.1)
    for step in range(5):
        g = np.array([0.5, -1.0, 2.0]) * (step + 1)
        opt.zero_grad(); w.grad = g.copy(); opt.step()
        opt_t.zero_grad(); wt.grad = torch.tensor(g); opt_t.step()
    assert np.allclose(w.data, wt.detach().numpy(), atol=1e-7)


def test_dataloader_batches_cover_everything_once():
    X, y = np.arange(10).reshape(10, 1), np.arange(10)
    dl = DataLoader(X, y, batch_size=4, shuffle=True, seed=0)
    assert len(dl) == 3
    batches = list(dl)
    assert [len(b[1]) for b in batches] == [4, 4, 2]
    seen = np.concatenate([b[1] for b in batches])
    assert sorted(seen) == list(range(10))
    assert not np.array_equal(seen, np.arange(10))    # actually shuffled
    assert all(np.array_equal(xb[:, 0], yb) for xb, yb in batches)   # X and y stay aligned


def test_dataloader_no_shuffle_keeps_order():
    X, y = np.arange(5).reshape(5, 1), np.arange(5)
    seen = np.concatenate([yb for _, yb in DataLoader(X, y, batch_size=2, shuffle=False)])
    assert np.array_equal(seen, np.arange(5))
