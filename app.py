"""
Streamlit app: classify clothing images with a CNN, explain them with Grad-CAM,
and log predictions to Supabase.

    streamlit run app.py
"""
from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path

import altair as alt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from src.data import CLASSES
from src.gradcam import GradCAM
from src.model import load_model, n_params
from src.preprocess import prepare
from src.supabase_log import SupabaseLog

ROOT = Path(__file__).parent
st.set_page_config(page_title="Fashion-MNIST CNN + Grad-CAM", page_icon="👕", layout="wide")


# ------------------------------------------------------------------ resources
@st.cache_resource
def get_model():
    model = load_model(str(ROOT / "models/fashion_cnn.pt"))
    return model, GradCAM(model, model.target_layer)


@st.cache_data
def get_metrics():
    p = ROOT / "results/metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


def get_db() -> SupabaseLog:
    try:
        url = st.secrets.get("SUPABASE_URL", "")
        key = st.secrets.get("SUPABASE_ANON_KEY", "")
    except Exception:  # no secrets file: run without Supabase
        url = key = ""
    return SupabaseLog(url, key)


def overlay(img28: np.ndarray, heat: np.ndarray) -> Image.Image:
    fig, ax = plt.subplots(figsize=(2.6, 2.6), dpi=110)
    ax.imshow(img28, cmap="gray")
    ax.imshow(heat, cmap="jet", alpha=0.45)
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return Image.open(buf)


model, cam = get_model()
metrics = get_metrics()
db = get_db()

st.title("👕 Fashion-MNIST CNN with Grad-CAM")
st.caption("A small PyTorch CNN classifies clothing images into 10 categories and shows where it "
           "looked. Predictions are logged to Supabase (no images stored).")

tab_cls, tab_res, tab_recent, tab_how = st.tabs(
    ["Classify", "Model results", "Recent predictions", "How it works"])

# ------------------------------------------------------------------ classify
with tab_cls:
    left, right = st.columns([1, 2])
    with left:
        mode = st.radio("Image source", ["Sample from the test set", "Upload your own"],
                        horizontal=False)
        img, source = None, None
        if mode.startswith("Sample"):
            samples = sorted((ROOT / "assets/samples").glob("*.png"))
            names = [f"{CLASSES[int(p.stem.split('_')[0])]} #{p.stem.rsplit('_', 1)[1]}"
                     for p in samples]
            pick = st.selectbox("Choose a held-out test image", range(len(samples)),
                                format_func=lambda i: names[i])
            img, source = Image.open(samples[pick]), "sample"
        else:
            up = st.file_uploader("PNG or JPG of one clothing item", type=["png", "jpg", "jpeg"])
            if up:
                img, source = Image.open(up), "upload"
        auto_invert = st.checkbox("Auto-invert light backgrounds", value=True,
                                  help="Fashion-MNIST items are light on a black background. "
                                       "Photos on white backgrounds are inverted to match.")

    if img is None:
        right.info("Pick a sample or upload an image.")
    else:
        raw = io.BytesIO()
        img.convert("RGB").save(raw, format="PNG")
        x, small, inverted = prepare(img, auto_invert)
        heat, probs = cam(torch.from_numpy(x))
        top = np.argsort(probs)[::-1][:5]
        pred = int(top[0])

        with right:
            c1, c2, c3 = st.columns(3)
            c1.image(img, caption="Your image", width="stretch")
            c2.image(Image.fromarray(small).resize((140, 140), Image.Resampling.NEAREST),
                     caption="What the model sees (28×28)" + (" · inverted" if inverted else ""))
            c3.image(overlay(small, heat), caption="Grad-CAM: where it looked")
            st.metric("Prediction", CLASSES[pred], f"{probs[pred]:.1%} confidence",
                      delta_color="off")
            top_df = pd.DataFrame({"class": [CLASSES[i] for i in top],
                                   "probability": [float(probs[i]) for i in top]})
            st.altair_chart(alt.Chart(top_df).mark_bar().encode(
                x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
                y=alt.Y("class:N", sort="-x", title=None),
                tooltip=["class", alt.Tooltip("probability:Q", format=".1%")]
            ).properties(height=170, title="Top-5 probabilities"), width="stretch")

            if db.enabled:
                key = hashlib.sha256(raw.getvalue()).hexdigest()
                if st.session_state.get("last_logged") != (key, auto_invert):
                    try:
                        db.insert("predictions", {
                            "image_sha256": key, "predicted_class": CLASSES[pred],
                            "confidence": float(probs[pred]),
                            "probabilities": {c: round(float(p), 5) for c, p in zip(CLASSES, probs)},
                            "source": source, "auto_inverted": bool(inverted),
                            "model_test_acc": metrics["test_acc"] if metrics else None})
                        st.session_state["last_logged"] = (key, auto_invert)
                        st.caption("✅ Logged to Supabase (hash, class and probabilities only).")
                    except Exception as e:
                        st.caption(f"⚠️ Could not log to Supabase: {e}")
                else:
                    st.caption("✅ Already logged.")
            else:
                st.caption("Supabase logging is off (no secrets configured).")

# ------------------------------------------------------------------ results
with tab_res:
    if not metrics:
        st.warning("No results/metrics.json yet. Run `python train.py` first.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test accuracy", f"{metrics['test_acc']:.2%}")
        c2.metric("Best validation accuracy", f"{metrics['best_val_acc']:.2%}")
        c3.metric("Parameters", f"{metrics['params']:,}")
        c4.metric("Epochs", metrics["settings"]["epochs"])

        hist = pd.DataFrame(metrics["history"])
        long = pd.concat([
            hist.assign(split="train", accuracy=hist.train_acc, loss=hist.train_loss),
            hist.assign(split="validation", accuracy=hist.val_acc, loss=hist.val_loss)])
        a, b = st.columns(2)
        for col, y in ((a, "loss"), (b, "accuracy")):
            col.altair_chart(alt.Chart(long).mark_line(point=True).encode(
                x="epoch:Q", y=alt.Y(f"{y}:Q", scale=alt.Scale(zero=False)), color="split:N",
                tooltip=["epoch", "split", alt.Tooltip(f"{y}:Q", format=".4f")]
            ).properties(title=y.capitalize(), height=260), width="stretch")

        cm = np.array(metrics["confusion_matrix"])
        cm_df = pd.DataFrame([{"true": CLASSES[i], "predicted": CLASSES[j], "count": int(cm[i, j])}
                              for i in range(10) for j in range(10)])
        base = alt.Chart(cm_df).encode(x=alt.X("predicted:N", sort=CLASSES),
                                       y=alt.Y("true:N", sort=CLASSES))
        heatmap = base.mark_rect().encode(color=alt.Color("count:Q", scale=alt.Scale(scheme="blues")),
                                          tooltip=["true", "predicted", "count"])
        text = base.mark_text(fontSize=10).encode(
            text="count:Q",
            color=alt.condition(alt.datum.count > cm.max() / 2, alt.value("white"), alt.value("black")))
        a, b = st.columns([3, 2])
        a.altair_chart((heatmap + text).properties(title="Confusion matrix (test set, 10,000 images)",
                                                   height=380), width="stretch")
        pc = pd.DataFrame(metrics["per_class"])
        b.dataframe(pc.style.format({"precision": "{:.1%}", "recall": "{:.1%}"}),
                    hide_index=True, width="stretch")
        worst = pc.sort_values("recall").iloc[0]
        b.caption(f"Hardest class: **{worst['class']}** (recall {worst['recall']:.1%}). "
                  "Shirts are often confused with T-shirts, pullovers and coats.")
        fig = ROOT / "results/figures/gradcam_examples.png"
        if fig.exists():
            st.image(str(fig), caption="Grad-CAM on one correctly classified test image per class")

# ------------------------------------------------------------------ recent
with tab_recent:
    if not db.enabled:
        st.info("Connect Supabase to see recent predictions (see README → Supabase setup).")
    else:
        try:
            rows = db.select("recent_predictions",
                             {"select": "*", "order": "created_at.desc", "limit": "200"})
        except Exception as e:
            rows = None
            st.error(f"Could not read from Supabase: {e}")
        if rows == []:
            st.info("No predictions logged yet. Classify an image first.")
        elif rows:
            df = pd.DataFrame(rows)
            df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime("%Y-%m-%d %H:%M")
            st.altair_chart(alt.Chart(df).mark_bar().encode(
                x=alt.X("predicted_class:N", sort=CLASSES, title=None), y="count():Q",
                color="source:N").properties(title=f"Last {len(df)} predictions", height=260),
                width="stretch")
            st.dataframe(df[["created_at", "predicted_class", "confidence", "source", "auto_inverted"]]
                         .style.format({"confidence": "{:.1%}"}),
                         hide_index=True, width="stretch")

# ------------------------------------------------------------------ how it works
with tab_how:
    st.markdown(f"""
**Data.** Fashion-MNIST: 70,000 grayscale 28×28 images of clothing in 10 balanced classes
(60,000 train, 10,000 test). 5,000 training images are held out for validation; the test set is
scored once, at the end.

**Model.** Three convolution blocks (3×3 convolutions with batch normalization and ReLU; max
pooling after the first two), global average pooling, dropout and one linear layer:
{n_params(model):,} parameters. Trained with AdamW, a one-cycle learning-rate schedule and
augmentation (random horizontal flips and shifts of up to 2 pixels).

**Grad-CAM.** For the predicted class, the gradient of its score with respect to the last
convolution block's feature maps tells how important each map is. A weighted sum of the maps,
passed through ReLU and upsampled to 28×28, gives the heatmap: red areas pushed the model
toward that class. It explains *this* model's decision, not what a person would look at.

**Uploads.** Images are converted to grayscale, inverted if the background is light (Fashion-MNIST
items are light on black), padded to a square and resized to 28×28. The model only knows
Fashion-MNIST-style images, so busy photos can still be misclassified with high confidence.

**Privacy.** Uploaded images are processed in memory and never stored. Supabase only receives a
SHA-256 hash of the image, the predicted class and the probabilities. The app uses the public
anon key, which can add predictions and read a hash-free view, nothing else.
""")
