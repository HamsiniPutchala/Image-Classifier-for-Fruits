"""Deployable Streamlit fruit classifier."""
from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = [BASE_DIR / "models" / "best_model.keras", BASE_DIR / "models" / "fruit_classifier.keras"]
CLASS_NAMES_PATH = BASE_DIR / "models" / "class_names.json"
DEFAULT_CLASSES = ["Apple", "Banana", "Blackberry", "Grape", "Kiwi", "Mango", "Orange", "Pineapple", "Strawberry", "Watermelon"]
IMAGE_SIZE = (128, 128)
UNKNOWN_THRESHOLD = 0.92
MIN_MARGIN = 0.22

st.set_page_config(page_title="Fruit Image Classifier", page_icon="🍎", layout="wide")


def load_classes() -> list[str]:
    if CLASS_NAMES_PATH.exists():
        try:
            data = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
            if isinstance(data, list) and len(data) == 10 and all(isinstance(x, str) for x in data):
                return data
        except (OSError, json.JSONDecodeError):
            pass
    return DEFAULT_CLASSES.copy()


@st.cache_resource(show_spinner="Loading MobileNetV2 model...")
def load_model(path: str) -> tf.keras.Model:
    return tf.keras.models.load_model(path)


def save_uploaded_model(data: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as f:
        f.write(data)
        return f.name


def find_model() -> Path | None:
    return next((p for p in MODEL_CANDIDATES if p.exists()), None)


def preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32) / 255.0


def quality(image: Image.Image) -> dict[str, Any]:
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    warnings: list[str] = []
    if float(gray.mean()) < 45: warnings.append("Image may be too dark")
    if float(gray.mean()) > 215: warnings.append("Image may be overexposed")
    if float(gray.std()) < 25: warnings.append("Low contrast")
    return {"quality": "Good" if not warnings else "Needs improvement", "warnings": warnings}


def predict(model: tf.keras.Model, image: Image.Image, classes: list[str]) -> dict[str, Any]:
    arr = preprocess(image)
    views = [arr, np.ascontiguousarray(arr[:, ::-1, :])]
    h, w = arr.shape[:2]
    crop = arr[int(h * .08):int(h * .92), int(w * .08):int(w * .92)]
    views.append(preprocess(Image.fromarray(np.clip(crop * 255, 0, 255).astype(np.uint8))))
    predictions = []
    for view in views:
        raw = np.asarray(model.predict(view[None, ...], verbose=0)[0], dtype=float)
        if len(raw) != len(classes): raise ValueError(f"Model has {len(raw)} outputs but class metadata has {len(classes)} classes.")
        if np.any(raw < 0) or not np.isclose(raw.sum(), 1.0, atol=.05): raw = np.exp(raw - raw.max())
        raw = raw / raw.sum()
        predictions.append(raw)
    probs = np.mean(predictions, axis=0)
    order = np.argsort(probs)[::-1]
    best = int(order[0])
    confidence = float(probs[best])
    margin = confidence - float(probs[int(order[1])])
    agreement = sum(int(np.argmax(p)) == best for p in predictions) / len(predictions)
    return {"class": classes[best], "confidence": confidence, "margin": margin, "agreement": agreement, "supported": confidence >= UNKNOWN_THRESHOLD and margin >= MIN_MARGIN and agreement >= 2 / 3, "top": [(classes[int(i)], float(probs[int(i)])) for i in order[:3]]}


def show_result(result: dict[str, Any], image_quality: dict[str, Any]) -> None:
    if not result["supported"]:
        st.error("🚫 No supported fruit could be identified confidently.")
        st.write(f"Top prediction: **{result['class']}** ({result['confidence'] * 100:.1f}%)")
        st.write(f"Prediction margin: {result['margin'] * 100:.1f}% | Multi-view agreement: {result['agreement'] * 100:.0f}%")
        st.info("Try a clear, well-lit image with one fruit centered in the frame.")
        return
    st.success(f"🍓 Predicted fruit: **{result['class']}**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Confidence", f"{result['confidence'] * 100:.2f}%")
    c2.metric("View agreement", f"{result['agreement'] * 100:.0f}%")
    c3.metric("Image quality", image_quality["quality"])
    for rank, (name, prob) in enumerate(result["top"], 1):
        st.write(f"**{rank}. {name}** — {prob * 100:.2f}%")
        st.progress(float(prob))
    if image_quality["warnings"]: st.warning(" • ".join(image_quality["warnings"]))


def model_setup_page(classes: list[str], bundled: Path | None) -> None:
    st.header("⚙️ Model Setup")
    if bundled:
        st.success(f"Bundled model found: `{bundled.relative_to(BASE_DIR)}`")
        return
    st.warning("No trained model is bundled in the repository. Upload the trained `.keras` model for this session.")
    uploaded = st.file_uploader("Upload the trained `.keras` model", type=["keras"], key="model_upload")
    if uploaded is None: return
    path = save_uploaded_model(uploaded.getvalue())
    try:
        model = load_model(path)
        outputs = int(model.output_shape[-1])
        if outputs != len(classes):
            st.error(f"Uploaded model has {outputs} outputs; this app requires {len(classes)} outputs.")
            return
        st.session_state["uploaded_model_path"] = path
        st.success("✅ Uploaded 10-class model is ready. You can now open Classifier or Camera.")
        st.json({"input_shape": str(model.input_shape), "output_shape": str(model.output_shape)})
    except Exception as exc:
        st.error(f"Could not load uploaded model: {exc}")


def home_page(classes: list[str], bundled: Path | None) -> None:
    """Modern landing page for the fruit classifier."""
    st.markdown(
        """
        <style>
        .hero {
            padding: 2.2rem 2.4rem;
            border-radius: 24px;
            background: linear-gradient(135deg, #fff4e6 0%, #fffaf4 48%, #eef9f1 100%);
            border: 1px solid rgba(0,0,0,.06);
            margin-bottom: 1.4rem;
        }
        .hero h1 { font-size: 3rem; margin: 0 0 .55rem 0; line-height: 1.05; }
        .hero p { font-size: 1.08rem; color: #4b5563; max-width: 760px; margin: 0; }
        .eyebrow { font-weight: 700; letter-spacing: .08em; text-transform: uppercase; font-size: .78rem; color: #d97706; margin-bottom: .65rem; }
        .card {
            padding: 1.15rem;
            border: 1px solid rgba(0,0,0,.08);
            border-radius: 18px;
            background: rgba(255,255,255,.72);
            min-height: 150px;
        }
        .card-title { font-weight: 700; font-size: 1.05rem; margin-bottom: .35rem; }
        .card-text { color: #5b6470; font-size: .92rem; line-height: 1.5; }
        .fruit-chip {
            display: inline-block;
            padding: .42rem .75rem;
            margin: .2rem .25rem .2rem 0;
            border-radius: 999px;
            background: #f7f7f7;
            border: 1px solid #e5e7eb;
            font-size: .88rem;
        }
        .section-label { font-size: 1.35rem; font-weight: 750; margin: 1.4rem 0 .7rem 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown(
            """
            <div class="hero">
                <div class="eyebrow">AI-Powered Fruit Recognition</div>
                <h1>🍎 Fruit Image Classifier</h1>
                <p>Identify fruits from an uploaded image or camera capture using a MobileNetV2 transfer-learning model with confidence and multi-view validation.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    m1, m2, m3, m4 = st.columns(4, gap="medium")
    m1.metric("🍓 Fruit classes", len(classes))
    m2.metric("🖼️ Input size", "128 × 128")
    m3.metric("🧠 Model", "MobileNetV2")
    m4.metric("🛡️ Views checked", "3")

    st.markdown('<div class="section-label">✨ What you can do</div>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3, gap="large")
    with f1:
        st.markdown('<div class="card"><div class="card-title">📸 Upload & Predict</div><div class="card-text">Upload JPG, JPEG or PNG images and receive the predicted fruit with confidence and top predictions.</div></div>', unsafe_allow_html=True)
    with f2:
        st.markdown('<div class="card"><div class="card-title">📷 Use Your Camera</div><div class="card-text">Capture a fruit directly from your device camera and run the same multi-view prediction pipeline.</div></div>', unsafe_allow_html=True)
    with f3:
        st.markdown('<div class="card"><div class="card-title">🛡️ Confidence-Aware Results</div><div class="card-text">Predictions are checked using confidence, prediction margin and agreement across multiple image views.</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">🍇 Supported fruits</div>', unsafe_allow_html=True)
    chips = "".join(f'<span class="fruit-chip">{name}</span>' for name in classes)
    st.markdown(chips, unsafe_allow_html=True)

    st.markdown('<div class="section-label">🚀 How it works</div>', unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4, gap="medium")
    for col, number, title, text in [
        (s1, "01", "Choose an image", "Upload an image or capture one with your camera."),
        (s2, "02", "Preprocess", "The image is resized and normalized for the model."),
        (s3, "03", "Analyze", "Three views are evaluated and their probabilities are combined."),
        (s4, "04", "Get the result", "See the fruit, confidence and ranked predictions."),
    ]:
        with col:
            st.markdown(f'<div class="card"><div class="eyebrow">{number}</div><div class="card-title">{title}</div><div class="card-text">{text}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">📦 Model status</div>', unsafe_allow_html=True)
    if bundled:
        st.success("✅ Trained 10-class model is bundled and ready. Open **Classifier** or **Camera** from the sidebar to start.")
    else:
        st.info("ℹ️ No bundled model was found. Open **Model Setup** to upload the trained `.keras` model for this session.")


def main() -> None:
    classes = load_classes()
    bundled = find_model()
    st.sidebar.title("🍎 Fruit Classifier")
    page = st.sidebar.radio("Navigate", ["🏠 Home", "📸 Classifier", "📷 Camera", "⚙️ Model Setup", "ℹ️ About"])
    st.sidebar.caption(f"{len(classes)} supported classes")
    st.sidebar.caption("MobileNetV2 • TensorFlow/Keras • Streamlit")

    if page == "🏠 Home":
        home_page(classes, bundled)
        return

    if page == "⚙️ Model Setup":
        model_setup_page(classes, bundled)
        return

    if page == "ℹ️ About":
        st.title("ℹ️ About")
        st.write("This project uses a MobileNetV2 transfer-learning model to classify ten fruit categories. Images are resized to 128×128 and evaluated across three views with confidence, margin and agreement gates.")
        st.write("**Classes:** " + ", ".join(classes))
        st.write("**Stack:** TensorFlow, Keras, Streamlit, NumPy, Pandas and Pillow")
        return

    model_path = bundled or st.session_state.get("uploaded_model_path")
    if model_path is None:
        st.warning("No trained model is available. Open **Model Setup** and upload the trained `.keras` model.")
        return
    try:
        model = load_model(str(model_path))
        if int(model.output_shape[-1]) != len(classes):
            st.error("Model/class metadata mismatch. The model must be trained for the current ten classes.")
            return
    except Exception as exc:
        st.error(f"Model loading failed: {exc}")
        return

    if page == "📸 Classifier":
        st.header("📸 Image Classification")
        files = st.file_uploader("Upload JPG/PNG images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        for file in files or []:
            image = Image.open(io.BytesIO(file.getvalue())).convert("RGB")
            st.image(image, caption=file.name, width=420)
            try: show_result(predict(model, image, classes), quality(image))
            except Exception as exc: st.error(f"Classification failed: {exc}")
    elif page == "📷 Camera":
        st.header("📷 Camera Classification")
        capture = st.camera_input("Capture a fruit")
        if capture:
            image = Image.open(io.BytesIO(capture.getvalue())).convert("RGB")
            try: show_result(predict(model, image, classes), quality(image))
            except Exception as exc: st.error(f"Camera prediction failed: {exc}")


if __name__ == "__main__":
    main()
