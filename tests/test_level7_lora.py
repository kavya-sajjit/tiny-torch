"""Level 7 (bonus): LoRA. A frozen base layer plus a trainable low-rank update."""
import numpy as np

from tinytorch.tensor import Tensor
from tinytorch.nn import Linear, mse_loss
from tinytorch.optim import Adam
from tinytorch.lora import LoRALinear

rng = np.random.default_rng(3)


def test_starts_identical_to_base():
    base = Linear(6, 4, seed=0)
    lora = LoRALinear(base, r=2, alpha=4, seed=0)
    x = Tensor(rng.normal(size=(3, 6)))
    assert np.allclose(lora(x).data, base(x).data)     # B starts at zero


def test_only_A_and_B_are_trainable():
    lora = LoRALinear(Linear(6, 4, seed=0), r=2, seed=0)
    params = lora.parameters()
    assert len(params) == 2
    assert {p.shape for p in params} == {(6, 2), (2, 4)}


def test_training_changes_output_but_not_base_weights():
    base = Linear(6, 4, seed=0)
    W_before = base.W.data.copy()
    lora = LoRALinear(base, r=2, alpha=4, seed=0)
    X, Y = rng.normal(size=(32, 6)), rng.normal(size=(32, 4))
    opt = Adam(lora.parameters(), lr=0.05)
    first = None
    for _ in range(100):
        opt.zero_grad()
        loss = mse_loss(lora(Tensor(X)), Y)
        first = first if first is not None else loss.data
        loss.backward()
        opt.step()
    assert loss.data < first
    assert np.array_equal(base.W.data, W_before)


def test_merge_gives_same_outputs():
    lora = LoRALinear(Linear(6, 4, seed=0), r=2, alpha=4, seed=0)
    lora.B.data = rng.normal(size=(2, 4))
    x = Tensor(rng.normal(size=(5, 6)))
    merged = lora.merge()
    assert np.allclose(merged(x).data, lora(x).data)
