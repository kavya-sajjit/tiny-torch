"""Level 5: feeding data in mini-batches."""
import numpy as np


class DataLoader:
    """Yields (X_batch, y_batch) pairs. Used as:  for xb, yb in DataLoader(X, y, 64): ..."""

    def __init__(self, X, y, batch_size=64, shuffle=True, seed=0):
        self.X, self.y = X, y
        self.batch_size, self.shuffle = batch_size, shuffle
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        # TODO(kavya): number of batches. 10 items in batches of 4 -> 3 (the last one is smaller).
        raise NotImplementedError("TODO(kavya): DataLoader.__len__")

    def __iter__(self):
        # TODO(kavya): this is a generator, so use `yield`.
        #   1. idx = np.arange(len(self.X)); shuffle it with self.rng if self.shuffle
        #   2. step through idx in chunks of batch_size and yield self.X[chunk], self.y[chunk]
        # Shuffle the INDICES, not X and y separately, or the labels won't match the inputs.
        raise NotImplementedError("TODO(kavya): DataLoader.__iter__")


# ---- given: you don't need to write this part --------------------------------
MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
MNIST_FILES = {
    "train_x": "train-images-idx3-ubyte.gz", "train_y": "train-labels-idx1-ubyte.gz",
    "test_x": "t10k-images-idx3-ubyte.gz", "test_y": "t10k-labels-idx1-ubyte.gz",
}


def load_mnist(root="data/mnist"):
    """Download MNIST once and return (X_train, y_train, X_test, y_test).

    X is float64 in [0, 1] with shape (N, 1, 28, 28); y is int64 with shape (N,).
    """
    import gzip
    import os
    import urllib.request

    os.makedirs(root, exist_ok=True)
    out = {}
    for key, name in MNIST_FILES.items():
        path = os.path.join(root, name)
        if not os.path.exists(path):
            print(f"downloading {name} ...")
            urllib.request.urlretrieve(MNIST_URL + name, path)
        with gzip.open(path, "rb") as f:
            raw = f.read()
        if key.endswith("_x"):
            out[key] = np.frombuffer(raw, np.uint8, offset=16).reshape(-1, 1, 28, 28) / 255.0
        else:
            out[key] = np.frombuffer(raw, np.uint8, offset=8).astype(np.int64)
    return out["train_x"], out["train_y"], out["test_x"], out["test_y"]
