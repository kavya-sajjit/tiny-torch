"""Level 6 goal: a small CNN built only from tinytorch reaches >= 98% on MNIST.

This is slow on CPU (NumPy, no GPU). Train on a 20k subset first to check it learns.
Run from the repo root:  python examples/train_mnist_cnn.py
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tinytorch.tensor import Tensor, cross_entropy
from tinytorch.nn import Linear, ReLU, Sequential
from tinytorch.conv import Conv2d, MaxPool2d, Flatten
from tinytorch.optim import Adam
from tinytorch.data import DataLoader, load_mnist


def main(epochs=3, n_train=20000):
    Xtr, ytr, Xte, yte = load_mnist()
    Xtr, ytr = Xtr[:n_train], ytr[:n_train]
    model = Sequential(
        Conv2d(1, 8, 3, padding=1, seed=0), ReLU(), MaxPool2d(2),     # 8 x 14 x 14
        Conv2d(8, 16, 3, padding=1, seed=1), ReLU(), MaxPool2d(2),    # 16 x 7 x 7
        Flatten(), Linear(16 * 7 * 7, 10, seed=2),
    )
    opt = Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        t0 = time.time()
        for xb, yb in DataLoader(Xtr, ytr, batch_size=64, seed=epoch):
            opt.zero_grad()
            loss = cross_entropy(model(Tensor(xb)), yb)
            loss.backward()
            opt.step()
        correct = sum((model(Tensor(Xte[s:s + 500])).data.argmax(1) == yte[s:s + 500]).sum()
                      for s in range(0, len(Xte), 500))
        print(f"epoch {epoch + 1}: test acc {correct / len(Xte):.4f}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
