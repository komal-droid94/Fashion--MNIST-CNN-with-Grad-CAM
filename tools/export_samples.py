"""
Save two held-out TEST images per class to assets/samples/ for the app's sample picker.

    python tools/export_samples.py
"""
from pathlib import Path
import sys

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import CLASSES, load  # noqa: E402

out = Path("assets/samples")
out.mkdir(parents=True, exist_ok=True)
_, _, x_test, y_test = load()
for c, name in enumerate(CLASSES):
    idx = [i for i, y in enumerate(y_test) if y == c][:2]
    for k, i in enumerate(idx, 1):
        slug = name.lower().replace("/", "-").replace(" ", "-")
        Image.fromarray(x_test[i]).resize((112, 112), Image.Resampling.NEAREST).save(
            out / f"{c}_{slug}_{k}.png")
print(f"saved {len(list(out.glob('*.png')))} images to {out}/")
