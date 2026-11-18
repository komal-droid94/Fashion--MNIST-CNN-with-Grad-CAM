"""
A compact CNN for 28x28 grayscale images.

Three convolution blocks (with batch normalization), global average pooling and one
linear layer. The last convolution block ("features.block3") is what Grad-CAM inspects.
"""
from __future__ import annotations
import torch
import torch.nn as nn


def conv_block(c_in: int, c_out: int, pool: bool) -> nn.Sequential:
    layers = [nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out),
              nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class FashionCNN(nn.Module):
    def __init__(self, n_classes: int = 10, width: int = 32, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential()
        self.features.add_module("block1", conv_block(1, width, pool=True))              # 28 -> 14
        self.features.add_module("block2", conv_block(width, 2 * width, pool=True))      # 14 -> 7
        self.features.add_module("block3", conv_block(2 * width, 4 * width, pool=False)) # 7 x 7
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(dropout), nn.Linear(4 * width, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))

    @property
    def target_layer(self) -> nn.Module:
        return self.features.block3


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(path: str, device: str | torch.device = "cpu") -> FashionCNN:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = FashionCNN(**ckpt.get("config", {}))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()
