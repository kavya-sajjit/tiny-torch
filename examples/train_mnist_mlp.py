"""Level 5 goal: a 2-layer MLP built only from tinytorch reaches >= 97% on MNIST.

Run from the repo root:  python examples/train_mnist_mlp.py
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tinytorch.tensor import Tensor, cross_entropy
from tinytorch.nn import Linear, ReLU, Sequential
from tinytorch.optim import Adam
from tinytorch.data import DataLoader, load_mnist


def accuracy(model, X, y, batch=1000):
    correct = 0
    for s in range(0, len(X), batch):
        logits = model(Tensor(X[s:s + batch].reshape(-1, 784)))
        correct += (logits.data.argmax(axis=1) == y[s:s + batch]).sum()
    return correct / len(X)


def main(epochs=5):
    Xtr, ytr, Xte, yte = load_mnist()
    model = Sequential(Linear(784, 256, seed=0), ReLU(), Linear(256, 10, seed=1))
    opt = Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        t0, losses = time.time(), []
        for xb, yb in DataLoader(Xtr, ytr, batch_size=128, seed=epoch):
            opt.zero_grad()
            loss = cross_entropy(model(Tensor(xb.reshape(-1, 784))), yb)
            loss.backward()
            opt.step()
            losses.append(loss.data)
        print(f"epoch {epoch + 1}: loss {np.mean(losses):.4f}  test acc {accuracy(model, Xte, yte):.4f}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
