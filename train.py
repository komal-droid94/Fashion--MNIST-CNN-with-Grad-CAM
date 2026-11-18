"""
Train the Fashion-MNIST CNN and save the model, metrics and figures.

    python train.py                       # 25 epochs, augmentation on (a few minutes on a GPU)
    python train.py --epochs 3            # quick check
    python train.py --log-supabase        # also record the run in Supabase (needs .env, see README)

Outputs:
    models/fashion_cnn.pt                 the trained model (best validation epoch)
    results/metrics.json                  settings, history, test accuracy, per-class metrics, confusion matrix
    results/figures/*.png                 training curves, confusion matrix, Grad-CAM examples
"""
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data import CLASSES, MEAN, STD, load, to_tensor_array, train_val_split
from src.gradcam import GradCAM
from src.model import FashionCNN, n_params, pick_device


def load_dotenv(path=".env"):
    """Tiny .env reader so no extra package is needed."""
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def augment(x: torch.Tensor, max_shift: float = 2 / 28) -> torch.Tensor:
    """Random horizontal flip and a random shift of up to 2 pixels, per image."""
    n = x.shape[0]
    flip = torch.rand(n, device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    shift = (torch.rand(n, 2, device=x.device) * 2 - 1) * max_shift * 2  # grid units span [-1, 1]
    theta = torch.zeros(n, 2, 3, device=x.device)
    theta[:, 0, 0] = theta[:, 1, 1] = 1
    theta[:, :, 2] = shift
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    background = -MEAN / STD  # value of a black pixel after normalization
    return F.grid_sample(x - background, grid, align_corners=False, padding_mode="zeros") + background


@torch.no_grad()
def evaluate(model, X, Y, device, bs=1000):
    model.eval()
    loss, preds = 0.0, []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(device)
        yb = torch.from_numpy(Y[i:i + bs]).long().to(device)
        out = model(xb)
        loss += F.cross_entropy(out, yb, reduction="sum").item()
        preds.append(out.argmax(1).cpu().numpy())
    preds = np.concatenate(preds)
    return loss / len(X), float((preds == Y).mean()), preds


def save_figures(history, cm, model, X_test, y_test, preds, device, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)

    ep = np.arange(1, len(history) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(ep, [h["train_loss"] for h in history], label="train")
    ax[0].plot(ep, [h["val_loss"] for h in history], label="validation")
    ax[0].set(title="Loss", xlabel="epoch")
    ax[1].plot(ep, [h["train_acc"] for h in history], label="train")
    ax[1].plot(ep, [h["val_acc"] for h in history], label="validation")
    ax[1].set(title="Accuracy", xlabel="epoch")
    for a in ax:
        a.legend(); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "training_curves.png", dpi=140); plt.close(fig)

    fig, a = plt.subplots(figsize=(7, 6))
    im = a.imshow(cm, cmap="Blues")
    a.set_xticks(range(10), CLASSES, rotation=45, ha="right"); a.set_yticks(range(10), CLASSES)
    a.set(xlabel="predicted", ylabel="true", title="Confusion matrix (test set)")
    for i in range(10):
        for j in range(10):
            a.text(j, i, cm[i][j], ha="center", va="center", fontsize=7,
                   color="white" if cm[i][j] > np.max(cm) / 2 else "black")
    fig.colorbar(im); fig.tight_layout(); fig.savefig(out / "confusion_matrix.png", dpi=140)
    plt.close(fig)

    # Grad-CAM: one correctly classified test image per class
    cam = GradCAM(model, model.target_layer)
    fig, axes = plt.subplots(2, 10, figsize=(15, 3.4))
    for c in range(10):
        i = int(np.flatnonzero((y_test == c) & (preds == c))[0])
        heat, probs = cam(torch.from_numpy(X_test[i:i + 1]).to(device))
        img = X_test[i, 0] * STD + MEAN
        axes[0, c].imshow(img, cmap="gray"); axes[0, c].set_title(CLASSES[c], fontsize=8)
        axes[1, c].imshow(img, cmap="gray"); axes[1, c].imshow(heat, cmap="jet", alpha=0.45)
        axes[1, c].set_title(f"{probs[c]:.0%}", fontsize=8)
        for a in axes[:, c]:
            a.axis("off")
    fig.suptitle("Grad-CAM on correctly classified test images (top: input, bottom: heatmap)")
    fig.tight_layout(); fig.savefig(out / "gradcam_examples.png", dpi=140); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--data", default="data")
    ap.add_argument("--log-supabase", action="store_true")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = pick_device()
    print(f"Training on: {device}", flush=True)

    x_tr, y_tr, x_te, y_te = load(a.data)
    x_tr, y_tr, x_va, y_va = train_val_split(x_tr, y_tr, seed=a.seed)
    Xtr, Xva, Xte = map(to_tensor_array, (x_tr, x_va, x_te))
    y_tr, y_va, y_te = (y.astype(np.int64) for y in (y_tr, y_va, y_te))
    print(f"train {len(Xtr)} · validation {len(Xva)} · test {len(Xte)}", flush=True)

    config = {"width": a.width, "dropout": a.dropout}
    model = FashionCNN(**config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * int(np.ceil(len(Xtr) / a.batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps)
    Xtr_t = torch.from_numpy(Xtr).to(device)
    ytr_t = torch.from_numpy(y_tr).to(device)

    history, best, best_state = [], -1.0, None
    t0 = time.time()
    for epoch in range(1, a.epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr_t), device=device)
        tot_loss, correct = 0.0, 0
        for i in range(0, len(perm), a.batch_size):
            idx = perm[i:i + a.batch_size]
            xb, yb = Xtr_t[idx], ytr_t[idx]
            if not a.no_augment:
                xb = augment(xb)
            out = model(xb)
            loss = F.cross_entropy(out, yb)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot_loss += loss.item() * len(idx)
            correct += (out.argmax(1) == yb).sum().item()
        val_loss, val_acc, _ = evaluate(model, Xva, y_va, device)
        h = {"epoch": epoch, "train_loss": tot_loss / len(Xtr), "train_acc": correct / len(Xtr),
             "val_loss": val_loss, "val_acc": val_acc}
        history.append(h)
        print(f"epoch {epoch:2d}  train loss {h['train_loss']:.3f} acc {h['train_acc']:.3f}  "
              f"val loss {val_loss:.3f} acc {val_acc:.3f}  ({time.time() - t0:.0f}s)", flush=True)
        if val_acc > best:
            best = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    _, test_acc, preds = evaluate(model, Xte, y_te, device)
    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(y_te, preds):
        cm[t, p] += 1
    per_class = []
    for c in range(10):
        tp = cm[c, c]
        per_class.append({"class": CLASSES[c],
                          "precision": float(tp / max(cm[:, c].sum(), 1)),
                          "recall": float(tp / max(cm[c, :].sum(), 1))})
    print(f"\nbest validation accuracy {best:.4f} · TEST ACCURACY {test_acc:.4f}", flush=True)

    Path("models").mkdir(exist_ok=True)
    torch.save({"state_dict": best_state, "config": config, "classes": CLASSES,
                "mean": MEAN, "std": STD}, "models/fashion_cnn.pt")
    metrics = {"settings": vars(a) | {"device": str(device)}, "params": n_params(model),
               "train_seconds": round(time.time() - t0, 1), "best_val_acc": best,
               "test_acc": test_acc, "per_class": per_class, "confusion_matrix": cm.tolist(),
               "history": history}
    Path("results").mkdir(exist_ok=True)
    Path("results/metrics.json").write_text(json.dumps(metrics, indent=2))
    save_figures(history, cm.tolist(), model, Xte, y_te, preds, device, Path("results/figures"))
    print("saved models/fashion_cnn.pt, results/metrics.json, results/figures/", flush=True)

    if a.log_supabase:
        from src.supabase_log import SupabaseLog
        load_dotenv()
        db = SupabaseLog(key=os.getenv("SUPABASE_SERVICE_KEY"))
        if not db.enabled:
            print("Supabase not configured (.env needs SUPABASE_URL and SUPABASE_SERVICE_KEY); skipped.")
        else:
            db.insert("runs", {"epochs": a.epochs, "batch_size": a.batch_size, "lr": a.lr,
                               "seed": a.seed, "augment": not a.no_augment,
                               "params": metrics["params"], "val_acc": best,
                               "test_acc": test_acc, "device": str(device)})
            print("run logged to Supabase table 'runs'")


if __name__ == "__main__":
    main()
