"""FruitVision AI - advanced Streamlit fruit classifier."""
from __future__ import annotations

import csv
import io
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

from utils import IMAGE_SIZE, get_class_names

BASE_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = [BASE_DIR / "models" / "best_model.keras", BASE_DIR / "models" / "fruit_classifier.keras"]
DATASET_PATH = BASE_DIR / "dataset"
UPLOADS_PATH = BASE_DIR / "uploads"
HISTORY_PATH = UPLOADS_PATH / "prediction_history.csv"
DEFAULT_CLASSES = ["Apple", "Banana", "Orange"]
UPLOADS_PATH.mkdir(exist_ok=True)

st.set_page_config(page_title="FruitVision AI", page_icon="🍎", layout="wide", initial_sidebar_state="expanded")


@st.cache_resource(show_spinner="Loading CNN model...")
def load_model_from_path(path: str) -> tf.keras.Model:
    return tf.keras.models.load_model(path)


def find_bundled_model() -> Path | None:
    return next((p for p in MODEL_CANDIDATES if p.exists()), None)


@st.cache_data
def load_classes() -> list[str]:
    try:
        train_path = DATASET_PATH / "train"
        if train_path.exists():
            classes = get_class_names(str(DATASET_PATH))
            if classes:
                return list(classes)
    except Exception:
        pass
    return DEFAULT_CLASSES.copy()


def preprocess_pil(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32) / 255.0


def confidence_label(confidence: float) -> str:
    if confidence >= 0.85:
        return "High confidence"
    if confidence >= 0.60:
        return "Moderate confidence"
    return "Low confidence — review image"


def image_quality(image: Image.Image) -> dict[str, Any]:
    rgb = image.convert("RGB")
    gray = np.asarray(rgb.convert("L"), dtype=np.float32)
    brightness = float(gray.mean())
    contrast = float(gray.std())
    sharpness = float(np.var(gray[:, 1:] - gray[:, :-1]))
    warnings: list[str] = []
    if brightness < 45:
        warnings.append("Image may be too dark")
    elif brightness > 215:
        warnings.append("Image may be overexposed")
    if contrast < 25:
        warnings.append("Low contrast")
    if sharpness < 80:
        warnings.append("Image may be blurry")
    return {"quality": "Good" if not warnings else "Needs improvement", "brightness": brightness, "contrast": contrast, "sharpness": sharpness, "warnings": warnings}


def predict_image(model: tf.keras.Model, image_array: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    raw = np.asarray(model.predict(np.expand_dims(image_array, axis=0), verbose=0)[0], dtype=float)
    if np.any(raw < 0) or not np.isclose(raw.sum(), 1.0, atol=0.05):
        exp = np.exp(raw - np.max(raw))
        probabilities = exp / exp.sum()
    else:
        probabilities = np.clip(raw, 0, None)
        probabilities /= probabilities.sum() if probabilities.sum() else 1
    if len(probabilities) != len(class_names):
        class_names = [f"Class {i + 1}" for i in range(len(probabilities))]
    order = np.argsort(probabilities)[::-1]
    top = [{"class": class_names[int(i)], "confidence": float(probabilities[int(i)]), "percentage": float(probabilities[int(i)] * 100)} for i in order[:min(3, len(order))]]
    best = top[0]
    return {"predicted_class": best["class"], "confidence": best["confidence"], "confidence_percentage": best["percentage"], "top_predictions": top, "probabilities": {class_names[int(i)]: float(probabilities[int(i)]) for i in order}}


def save_prediction_history(image_name: str, predicted_class: str, confidence: float, method: str, quality: str) -> None:
    exists = HISTORY_PATH.exists()
    with HISTORY_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not exists:
            writer.writerow(["Timestamp", "Image Name", "Predicted Class", "Confidence (%)", "Method", "Image Quality"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), image_name, predicted_class, f"{confidence:.2f}", method, quality])


def show_prediction(results: dict[str, Any], quality: dict[str, Any]) -> None:
    confidence = results["confidence"]
    if confidence >= 0.60:
        st.success(f"Predicted fruit: **{results['predicted_class']}**")
    else:
        st.warning(f"Low-confidence prediction: **{results['predicted_class']}**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Confidence", f"{results['confidence_percentage']:.2f}%")
    c2.metric("Assessment", confidence_label(confidence))
    c3.metric("Image quality", quality["quality"])
    st.progress(confidence, text=f"Top prediction confidence: {results['confidence_percentage']:.1f}%")
    if quality["warnings"]:
        st.caption(" • ".join(quality["warnings"]))
    st.markdown("#### Top predictions")
    for rank, item in enumerate(results["top_predictions"], start=1):
        st.write(f"**{rank}. {item['class']}** — {item['percentage']:.2f}%")
        st.progress(item["confidence"])
    probability_df = pd.DataFrame({"Fruit": list(results["probabilities"].keys()), "Probability (%)": [v * 100 for v in results["probabilities"].values()]})
    st.bar_chart(probability_df.set_index("Fruit"))


def classify_bytes(model: tf.keras.Model, data: bytes, classes: list[str]) -> tuple[Image.Image, dict[str, Any], dict[str, Any]]:
    image = Image.open(io.BytesIO(data)).convert("RGB")
    quality = image_quality(image)
    results = predict_image(model, preprocess_pil(image), classes)
    return image, results, quality


def upload_page(model: tf.keras.Model, classes: list[str]) -> None:
    st.header("📸 AI Fruit Classifier")
    st.caption("Batch classification with confidence, image-quality checks and ranked predictions.")
    files = st.file_uploader("Upload fruit images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if not files:
        st.info("Upload one or more JPG/PNG images to begin.")
        return
    all_rows = []
    for uploaded_file in files:
        st.divider()
        left, right = st.columns([1, 1.35])
        data = uploaded_file.getvalue()
        with left:
            image = Image.open(io.BytesIO(data)).convert("RGB")
            st.image(image, caption=uploaded_file.name, use_container_width=True)
            st.caption(f"{image.width} × {image.height}px")
        with right:
            try:
                _, results, quality = classify_bytes(model, data, classes)
                show_prediction(results, quality)
                save_prediction_history(uploaded_file.name, results["predicted_class"], results["confidence_percentage"], "upload", quality["quality"])
                all_rows.append({"Image": uploaded_file.name, "Prediction": results["predicted_class"], "Confidence (%)": round(results["confidence_percentage"], 2), "Quality": quality["quality"]})
            except Exception as exc:
                st.error(f"Could not classify {uploaded_file.name}: {exc}")
    if all_rows:
        st.subheader("Batch results")
        batch_df = pd.DataFrame(all_rows)
        st.dataframe(batch_df, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download batch results", batch_df.to_csv(index=False).encode("utf-8"), "fruit_batch_results.csv", "text/csv")


def camera_page(model: tf.keras.Model, classes: list[str]) -> None:
    st.header("📷 Camera Classification")
    camera_image = st.camera_input("Capture a fruit")
    if camera_image is None:
        st.info("Allow camera access and capture an image.")
        return
    try:
        image, results, quality = classify_bytes(model, camera_image.getvalue(), classes)
        st.image(image, caption="Captured fruit", width=450)
        show_prediction(results, quality)
        save_prediction_history("camera_capture.jpg", results["predicted_class"], results["confidence_percentage"], "camera", quality["quality"])
    except Exception as exc:
        st.error(f"Camera prediction failed: {exc}")


def dashboard_page() -> None:
    st.header("📊 Analytics Dashboard")
    if not HISTORY_PATH.exists():
        st.info("No predictions recorded yet. Classify an image first.")
        return
    df = pd.read_csv(HISTORY_PATH)
    if df.empty:
        st.info("No prediction records available.")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total predictions", len(df))
    c2.metric("Most predicted", str(df["Predicted Class"].mode().iloc[0]))
    c3.metric("Average confidence", f"{df['Confidence (%)'].mean():.1f}%")
    c4.metric("High confidence", f"{(df['Confidence (%)'] >= 85).mean() * 100:.1f}%")
    st.divider()
    st.subheader("Prediction distribution")
    st.bar_chart(df["Predicted Class"].value_counts().rename("Predictions"))
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Export complete history", df.to_csv(index=False).encode("utf-8"), "fruit_prediction_history.csv", "text/csv")
    if st.button("🗑️ Clear history"):
        HISTORY_PATH.unlink(missing_ok=True)
        st.rerun()


def model_page() -> tf.keras.Model | None:
    bundled = find_bundled_model()
    if bundled:
        try:
            return load_model_from_path(str(bundled))
        except Exception as exc:
            st.error(f"Bundled model could not be loaded: {exc}")
    st.warning("No trained CNN model is bundled with this repository yet.")
    st.markdown("### Model setup")
    st.write("Upload your trained `.keras` model below. For permanent deployment, add the trained model as `models/best_model.keras` to the repository.")
    model_file = st.file_uploader("Upload trained Keras model", type=["keras"], key="model_uploader")
    if model_file:
        try:
            with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
                tmp.write(model_file.getvalue())
                path = tmp.name
            return load_model_from_path(path)
        except Exception as exc:
            st.error(f"Model upload failed: {exc}")
    return None


def about_page(classes: list[str]) -> None:
    st.header("ℹ️ About FruitVision AI")
    st.markdown("**FruitVision AI** is an internship-ready CNN application for fruit recognition with batch inference, camera input, confidence scoring, image-quality analysis, analytics and downloadable reports.")
    st.subheader("Supported classes")
    st.write(" • ".join(classes))
    st.subheader("Recommended image quality")
    st.write("Use one clearly visible fruit, good lighting, moderate background clutter and a sharp image for better predictions.")


def main() -> None:
    st.markdown("<style>.block-container{padding-top:2rem;padding-bottom:3rem} [data-testid=stMetricValue]{font-size:1.6rem}</style>", unsafe_allow_html=True)
    st.title("🍎 FruitVision AI")
    st.caption("Advanced CNN-powered Fruit Image Classification System")
    classes = load_classes()
    with st.sidebar:
        st.title("Navigation")
        page = st.radio("Choose a module", ["📸 Classifier", "📷 Camera", "📊 Dashboard", "⚙️ Model Setup", "ℹ️ About"])
        st.divider()
        st.info(f"Classes: {len(classes)}")
        st.caption("TensorFlow / Keras CNN inference")
    if page == "⚙️ Model Setup":
        model_page()
        return
    model = model_page()
    if model is None:
        st.stop()
    if page == "📸 Classifier":
        upload_page(model, classes)
    elif page == "📷 Camera":
        camera_page(model, classes)
    elif page == "📊 Dashboard":
        dashboard_page()
    else:
        about_page(classes)


if __name__ == "__main__":
    main()
