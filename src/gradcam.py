"""
Grad-CAM (Selvaraju et al., 2017): which parts of the image pushed the model toward a class.

1. Run the image forward and keep the target layer's activations A (C x H x W).
2. Backpropagate the chosen class score and keep the gradients dS/dA.
3. Weight each channel by its average gradient, sum over channels, apply ReLU.
4. Upsample the 7x7 map to 28x28 and scale it to 0-1.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self._acts = None
        self._grads = None
        target_layer.register_forward_hook(self._save_acts)
        target_layer.register_full_backward_hook(self._save_grads)

    def _save_acts(self, module, inputs, output):
        self._acts = output.detach()

    def _save_grads(self, module, grad_in, grad_out):
        self._grads = grad_out[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None):
        """x: (1, 1, 28, 28). Returns (heatmap 28x28 numpy in [0,1], probabilities numpy)."""
        self.model.eval()
        x = x.clone().requires_grad_(True)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0]
        if class_idx is None:
            class_idx = int(probs.argmax())
        self.model.zero_grad()
        logits[0, class_idx].backward()
        weights = self._grads.mean(dim=(2, 3), keepdim=True)          # (1, C, 1, 1)
        cam = F.relu((weights * self._acts).sum(dim=1, keepdim=True))  # (1, 1, 7, 7)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.cpu().numpy(), probs.detach().cpu().numpy()
