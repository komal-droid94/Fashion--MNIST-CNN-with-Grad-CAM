"""
Fashion-MNIST: download, verify, load, and split.

The four official files are downloaded once into ./data (git-ignored) and checked against
their published MD5 hashes. Images are 28x28 grayscale, labels 0-9.
"""
from __future__ import annotations
import gzip
import hashlib
import urllib.request
from pathlib import Path

import numpy as np

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Dataset statistics of the training images (pixel values scaled to 0-1).
MEAN, STD = 0.2860, 0.3530

FILES = {
    "train_images": ("train-images-idx3-ubyte.gz", "8d4fb7e6c68d591d4c3dfef9ec88bf0d"),
    "train_labels": ("train-labels-idx1-ubyte.gz", "25c81989df183df01b3e8a0aad5dffbe"),
    "test_images": ("t10k-images-idx3-ubyte.gz", "bef4ecab320f06d8554ea6380940ec79"),
    "test_labels": ("t10k-labels-idx1-ubyte.gz", "bb300cfdad3c16e7a12a480ee83cd310"),
}
MIRRORS = [
    "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/",
    "https://ossci-datasets.s3.amazonaws.com/fashion-mnist/",
    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
]


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def download(data_dir: str | Path = "data") -> Path:
    """Download the four files if missing and verify their MD5 hashes."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, md5 in FILES.values():
        path = data_dir / name
        if path.exists() and _md5(path) == md5:
            continue
        for base in MIRRORS:
            try:
                print(f"Downloading {name} from {base} ...", flush=True)
                urllib.request.urlretrieve(base + name, path)
                if _md5(path) == md5:
                    break
                print("  checksum mismatch, trying next mirror")
            except Exception as e:  # network errors: try the next mirror
                print(f"  failed: {e}")
        else:
            raise RuntimeError(f"Could not download {name} from any mirror.")
    return data_dir


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        raw = f.read()
    magic = int.from_bytes(raw[2:4], "big")
    ndim = raw[3]
    dims = [int.from_bytes(raw[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim)]
    del magic
    return np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


def load(data_dir: str | Path = "data"):
    """Return (x_train, y_train, x_test, y_test); images as uint8 arrays (N, 28, 28)."""
    data_dir = download(data_dir)
    get = lambda key: _read_idx(data_dir / FILES[key][0])
    return get("train_images"), get("train_labels"), get("test_images"), get("test_labels")


def train_val_split(x, y, val_size: int = 5000, seed: int = 0):
    """Hold out a stratified validation set from the training images."""
    rng = np.random.default_rng(seed)
    per_class = val_size // len(CLASSES)
    val_idx = np.concatenate([rng.choice(np.flatnonzero(y == c), per_class, replace=False)
                              for c in range(len(CLASSES))])
    mask = np.ones(len(y), dtype=bool)
    mask[val_idx] = False
    return x[mask], y[mask], x[val_idx], y[val_idx]


def to_tensor_array(x: np.ndarray) -> np.ndarray:
    """uint8 (N, 28, 28) -> normalized float32 (N, 1, 28, 28)."""
    return ((x.astype(np.float32) / 255.0 - MEAN) / STD)[:, None, :, :]
