"""
Turn any uploaded picture into the 28x28 format the model was trained on.

Fashion-MNIST items are light garments centred on a black background. Ordinary photos are
usually the opposite (dark item, light background), which fools the model. So we:
  1. convert to grayscale,
  2. invert if the border is brighter than the centre (light background),
  3. pad to a square so the aspect ratio is kept, then resize to 28x28,
  4. normalize with the training mean and standard deviation.
"""
from __future__ import annotations
import numpy as np
from PIL import Image, ImageOps

from .data import MEAN, STD


def has_light_background(gray: np.ndarray) -> bool:
    """True if the image's border is brighter than its centre."""
    h, w = gray.shape
    b = max(1, min(h, w) // 10)
    border = np.concatenate([gray[:b].ravel(), gray[-b:].ravel(), gray[:, :b].ravel(),
                             gray[:, -b:].ravel()])
    centre = gray[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    return border.mean() > centre.mean()


def prepare(img: Image.Image, auto_invert: bool = True):
    """Returns (model_input float32 (1,1,28,28), display image uint8 (28,28), inverted flag)."""
    gray = ImageOps.exif_transpose(img).convert("L")
    arr = np.asarray(gray)
    inverted = auto_invert and has_light_background(arr)
    if inverted:
        gray = ImageOps.invert(gray)
    side = max(gray.size)
    square = Image.new("L", (side, side), 0)
    square.paste(gray, ((side - gray.size[0]) // 2, (side - gray.size[1]) // 2))
    small = np.asarray(square.resize((28, 28), Image.Resampling.LANCZOS), dtype=np.uint8)
    x = ((small.astype(np.float32) / 255.0 - MEAN) / STD)[None, None]
    return x, small, inverted
