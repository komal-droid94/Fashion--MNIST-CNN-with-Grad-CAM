import numpy as np
import torch
from PIL import Image

from src.data import CLASSES, to_tensor_array, train_val_split
from src.gradcam import GradCAM
from src.model import FashionCNN
from src.preprocess import has_light_background, prepare
from src.supabase_log import SupabaseLog


def test_model_output_shape():
    m = FashionCNN().eval()
    assert m(torch.zeros(4, 1, 28, 28)).shape == (4, len(CLASSES))


def test_gradcam_shape_and_range():
    m = FashionCNN()
    heat, probs = GradCAM(m, m.target_layer)(torch.randn(1, 1, 28, 28))
    assert heat.shape == (28, 28)
    assert heat.min() >= 0 and heat.max() <= 1
    assert abs(probs.sum() - 1) < 1e-5


def test_preprocess_inverts_light_background():
    img = np.full((60, 40), 240, np.uint8)
    img[15:45, 10:30] = 30                      # dark item on white background
    x, small, inverted = prepare(Image.fromarray(img))
    assert inverted and x.shape == (1, 1, 28, 28) and small.shape == (28, 28)
    assert small[0, 0] < 50 and small[14, 14] > 150   # now light item on dark background


def test_preprocess_keeps_dark_background():
    img = np.zeros((28, 28), np.uint8)
    img[8:20, 8:20] = 200
    assert not has_light_background(img)
    _, small, inverted = prepare(Image.fromarray(img))
    assert not inverted and small[14, 14] > 150


def test_split_is_stratified_and_disjoint():
    x = np.arange(1000).reshape(1000, 1, 1).repeat(28, 1).repeat(28, 2).astype(np.uint8)
    y = np.repeat(np.arange(10), 100)
    xt, yt, xv, yv = train_val_split(x, y, val_size=200)
    assert len(xv) == 200 and np.all(np.bincount(yv) == 20)
    assert len(xt) + len(xv) == 1000
    assert to_tensor_array(xt).shape == (800, 1, 28, 28)


def test_supabase_disabled_without_config(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    assert not SupabaseLog().enabled
