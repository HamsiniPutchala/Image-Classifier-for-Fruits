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
DEFAULT_CLASSES = [
    "Apple", "Banana", "Blackberry", "Grape", "Kiwi",
    "Mango", "Orange", "Pineapple", "Strawberry", "Watermelon",
]
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


def save_uploaded_model(data: bytes, suffix: str = ".keras") -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
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
    if float(gray.mean()) < 45:
        warnings.append("Image may be too dark")
    if float(gray.mean()) > 215:
        warnings.append("Image may be overexposed")
    if float(gray.std()) < 25:
        warnings.append("Low contrast")
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
        if len(raw) != len(classes):
            raise ValueError(f"Model has {len(raw)} outputs but class metadata has {len(classes)} classes.")
        if np.any(raw < 0) or not np.isclose(raw.sum(), 1.0, atol=.05):
            raw = np.exp(raw - raw.max())
        raw = raw / raw.sum()
        predictions.append(raw)

    probs = np.mean(predictions, axis=0)
    order = np.argsort(probs)[::-1]
    best = int(order[0])
    second = float(probs[int(order[1])])
    confidence = float(probs[best])
    margin = confidence - second
    agreement = sum(int(np.argmax(p)) == best for p in predictions) / len(predictions)
    supported = confidence >= UNKNOWN_THRESHOLD and margin >= MIN_MARGIN and agreement >= 2 / 3

    return {
        "class": classes[best],
        "confidence": confidence,
        "margin": margin,
        "agreement": agreement,
        "supported": supported,
        "top": [(classes[int(i)], float(probs[int(i)])) for i in order[:3]],
    }


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
    if image_quality["warnings"]:
        st.warning(" • ".join(image_quality["warnings"]))


def model_setup_page(classes: list[str], bundled: Path | None) -> str | None:
    st.header("⚙️ Model Setup")
    st.write("The deployed app can use a model bundled in `models/` or a temporary `.keras` upload. This makes the Streamlit deployment start even when the model is too large to commit to GitHub.")
    if bundled:
        st.success(f"Bundled model found: `{bundled.relative_to(BASE_DIR)}`")
        return str(bundled)
    st.warning("No bundled model is present in the repository.")
    uploaded = st.file_uploader("Upload the trained `.keras` model", type=["keras"], key="model_upload")
    if uploaded is not None:
        path = save_uploaded_model(uploaded.getvalue())
        try:
            model = load_model(path)
            outputs = int(model.output_shape[-1])
            if outputs != len(classes):
                st.error(f"Uploaded model has {outputs} outputs; this app requires {len(classes)} outputs.")
                return None
            st.success("Uploaded 10-class model is ready for this session.")
            return path
        except Exception as exc:
            st.error(f"Could not load uploaded model: {exc}")
    return None


def main() -> None:
    classes = load_classes()
    bundled = find_model()
    st.sidebar.title("🍎 Fruit Classifier")
    page = st.sidebar.radio("Navigate", ["🏠 Home", "📸 Classifier", "📷 Camera", "⚙️ Model Setup", "ℹ️ About"])
    st.sidebar.caption(f"{len(classes)} supported classes")
    st.sidebar.caption("MobileNetV2 • TensorFlow/Keras • Streamlit")

    if page == "🏠 Home":
        st.title("🍎 Image Classifier for Fruits")
        st.write("An interactive fruit-recognition application powered by MobileNetV2 transfer learning.")
        st.metric("Supported fruits", len(classes))
        st.subheader("Supported classes")
        st.write(" • ".join(classes))
        if bundled:
            st.success("Trained model is bundled with the app.")
        else:
            st.info("The app is deployable now. Upload the trained `.keras` model from Model Setup before making predictions.")
        return

    if page == "⚙️ Model Setup":
        model_path = model_setup_page(classes, bundled)
        if model_path:
            try:
                model = load_model(model_path)
                st.json({"input_shape": str(model.input_shape), "output_shape": str(model.output_shape), "classes": classes})
            except Exception as exc:
                st.error(str(exc))
        return

    if page == "ℹ️ About":
        st.title("ℹ️ About")
        st.write("This project uses a MobileNetV2 transfer-learning model to classify ten fruit categories. Images are resized to 128×128, evaluated across three views, and accepted only when confidence, margin and view agreement are sufficiently strong.")
        st.write("**Classes:** " + ", ".join(classes))
        st.write("**Stack:** TensorFlow, Keras, Streamlit, NumPy, Pandas and Pillow")
        return

    model_path = bundled
    if model_path is None:
        model_path = st.session_state.get("uploaded_model_path")
    if model_path is None:
        st.warning("No trained model is available. Open **Model Setup** and upload the trained `.keras` model.")
        return

    try:
        model = load_model(model_path)
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
            try:
                result = predict(model, image, classes)
                show_result(result, quality(image))
            except Exception as exc:
                st.error(f"Classification failed: {exc}")

    elif page == "📷 Camera":
        st.header("📷 Camera Classification")
        capture = st.camera_input("Capture a fruit")
        if capture:
            image = Image.open(io.BytesIO(capture.getvalue())).convert("RGB")
            result = predict(model, image, classes)
            show_result(result, quality(image))


if __name__ == "__main__":
    main()
