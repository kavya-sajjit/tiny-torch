# tiny-torch

A small deep-learning library written from scratch in NumPy: automatic differentiation, layers, optimisers, and a CNN that trains on MNIST. It is built to understand how PyTorch works underneath, one level at a time.

Every gradient is checked against real PyTorch in the test suite.

## How this repo works

The code is a **starter kit**. Every function you need to write is a stub marked `# TODO(kavya)` that raises `NotImplementedError`. The tests already exist and fail. Your job is to make them pass, one level at a time.

```bash
pip install -r requirements.txt
pytest tests/test_level1_value.py -q      # run only the level you are on
pytest -q                                 # run everything
```

**Rules for yourself**

1. Turn off Copilot / AI autocomplete for files under `tinytorch/`. Struggling is the learning.
2. Stuck for more than 20 minutes? Read the hint in the docstring, then ask for the *next* hint, not the answer.
3. After each level, write 3–5 lines in [`LEARNING.md`](LEARNING.md): what broke, what you learned.
4. One branch and one PR per level (`level-1`, `level-2`, ...). Every PR gets reviewed.

## Levels

| Level | File | What you build | Done when |
|---|---|---|---|
| **1. Scalar autograd** | `tinytorch/value.py` | A `Value` class holding one number. `+`, `*`, `**`, `relu`, `tanh`, `exp`, and `backward()` using a topological sort | `pytest tests/test_level1_value.py` passes |
| **2. Tensors and broadcasting** | `tinytorch/tensor.py` | The same idea on NumPy arrays: elementwise ops, and `unbroadcast` so gradients have the right shape | `test_level2_tensor.py` passes |
| **3. Reductions and matmul** | `tinytorch/tensor.py` | `sum`, `mean`, `matmul`, `reshape`, `log`, and a numerically stable `cross_entropy` | `test_level3_ops.py` passes |
| **4. Layers and SGD** | `tinytorch/nn.py`, `tinytorch/optim.py` | `Module`, `Linear`, `ReLU`, `Sequential`, `SGD`. A tiny MLP learns XOR | `test_level4_nn.py` passes |
| **5. Adam and MNIST** | `tinytorch/optim.py`, `tinytorch/data.py` | `Adam`, a shuffling `DataLoader`. `examples/train_mnist_mlp.py` reaches **≥97%** test accuracy | tests pass and the script prints ≥0.97 |
| **6. Convolutions** | `tinytorch/conv.py` | `Conv2d` via im2col, `MaxPool2d`, `Flatten`. A small CNN on MNIST reaches **≥98%** | `test_level6_conv.py` passes |
| **7. Bonus: LoRA** | `tinytorch/lora.py` | A `LoRALinear` layer: frozen base weight + trainable low-rank `B @ A`. Bridges into the LoRA paper project | `test_level7_lora.py` passes |

## Reading list (one per level, read *before* you start it)

1. Karpathy, *The spelled-out intro to neural networks and backpropagation* (micrograd video). Watch once, then build Level 1 **without** looking.
2. NumPy docs: [Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)
3. CS231n notes: [Backprop, intuitions](https://cs231n.github.io/optimization-2/) and the softmax section of [Linear classification](https://cs231n.github.io/linear-classify/)
4. PyTorch docs: [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)
5. Kingma & Ba (2014), *Adam*, Algorithm 1 only
6. CS231n notes: [Convolutional networks](https://cs231n.github.io/convolutional-networks/), the "im2col" paragraph
7. Hu et al. (2021), *LoRA*, sections 1–4

## Layout

```
tinytorch/
  value.py     Level 1   scalar autograd
  tensor.py    Level 2-3 Tensor + ops
  nn.py        Level 4   Module, Linear, ReLU, Sequential, losses
  optim.py     Level 4-5 SGD, Adam
  data.py      Level 5   DataLoader, MNIST loader (given)
  conv.py      Level 6   Conv2d, MaxPool2d, Flatten
  lora.py      Level 7   LoRALinear
tests/         one file per level; gradients compared to PyTorch
examples/      training scripts
LEARNING.md    your log
```
