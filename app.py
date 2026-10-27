"""Advanced Streamlit application for fruit image classification."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

from utils import IMAGE_SIZE, get_class_names, load_and_preprocess_image

BASE_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = [BASE_DIR / "models" / "best_model.keras", BASE_DIR / "models" / "fruit_classifier.keras"]
DATASET_PATH = BASE_DIR / "dataset"
UPLOADS_PATH = BASE_DIR / "uploads"
HISTORY_PATH = UPLOADS_PATH / "prediction_history.csv"
DEFAULT_CLASSES = ["Apple", "Banana", "Orange"]
UPLOADS_PATH.mkdir(exist_ok=True)

st.set_page_config(page_title="FruitVision AI", page_icon="🍎", layout="wide", initial_sidebar_state="expanded")


@st.cache_resource(show_spinner="Loading the CNN model...")
def load_model() -> tf.keras.Model:
    for path in MODEL_CANDIDATES:
        if path.exists():
            return tf.keras.models.load_model(path)
    raise FileNotFoundError("No trained model was found. Add best_model.keras or fruit_classifier.keras to models/.")


@st.cache_data
def load_classes() -> list[str]:
    try:
        if DATASET_PATH.exists():
            classes = get_class_names(str(DATASET_PATH))
            if classes:
                return list(classes)
    except Exception:
        pass
    return DEFAULT_CLASSES.copy()


def predict_image(model: tf.keras.Model, image_array: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    probabilities = np.asarray(model.predict(np.expand_dims(image_array, axis=0), verbose=0)[0], dtype=float)
    if len(probabilities) != len(class_names):
        class_names = [f"Class {i}" for i in range(len(probabilities))]
    probabilities = np.clip(probabilities, 0, None)
    total = probabilities.sum()
    if total > 0:
        probabilities /= total
    order = np.argsort(probabilities)[::-1]
    top = [{"class": class_names[int(i)], "confidence": float(probabilities[int(i)]), "percentage": float(probabilities[int(i)] * 100)} for i in order[:min(3, len(order))]]
    best = top[0]
    return {
        "predicted_class": best["class"],
        "confidence": best["confidence"],
        "confidence_percentage": best["percentage"],
        "top_predictions": top,
        "probabilities": {class_names[int(i)]: float(probabilities[int(i)]) for i in order},
    }


def confidence_label(confidence: float) -> str:
    if confidence >= 0.85:
        return "High confidence"
    if confidence >= 0.60:
        return "Moderate confidence"
    return "Low confidence"


def save_prediction_history(image_name: str, predicted_class: str, confidence: float, method: str) -> None:
    exists = HISTORY_PATH.exists()
    with HISTORY_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not exists:
            writer.writerow(["Timestamp", "Image Name", "Predicted Class", "Confidence (%)", "Method"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), image_name, predicted_class, f"{confidence:.2f}", method])


def show_prediction(results: dict[str, Any]) -> None:
    confidence = results["confidence"]
    st.success(f"Predicted fruit: **{results['predicted_class']}**")
    col1, col2 = st.columns(2)
    col1.metric("Confidence", f"{results['confidence_percentage']:.2f}%")
    col2.metric("Assessment", confidence_label(confidence))
    st.progress(confidence, text=f"Model confidence: {results['confidence_percentage']:.1f}%")
    st.markdown("#### Top predictions")
    for rank, item in enumerate(results["top_predictions"], start=1):
        st.write(f"**{rank}. {item['class']}** — {item['percentage']:.2f}%")
        st.progress(item["confidence"])
    probability_df = pd.DataFrame({"Fruit": list(results["probabilities"].keys()), "Probability (%)": [v * 100 for v in results["probabilities"].values()]})
    st.bar_chart(probability_df.set_index("Fruit"))


def upload_page(model: tf.keras.Model, classes: list[str]) -> None:
    st.header("📸 Classify Fruit Images")
    st.caption("Upload one or more fruit images and get CNN predictions with confidence scores.")
    files = st.file_uploader("Choose JPG, JPEG or PNG images", type=["jpg", "jpeg", "png"], accept_multiple_files=True, help="For best results, use a clear image containing one fruit.")
    if not files:
        st.info("Upload an image to start the prediction.")
        return
    for uploaded_file in files:
        st.divider()
        left, right = st.columns([1, 1.15])
        with left:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption=uploaded_file.name, use_container_width=True)
            st.caption(f"Image size: {image.width} × {image.height}px")
        with right:
            try:
                results = predict_image(model, load_and_preprocess_image(uploaded_file, IMAGE_SIZE), classes)
                show_prediction(results)
                save_prediction_history(uploaded_file.name, results["predicted_class"], results["confidence_percentage"], "upload")
            except Exception as exc:
                st.error(f"Unable to classify this image: {exc}")


def camera_page(model: tf.keras.Model, classes: list[str]) -> None:
    st.header("📷 Camera Prediction")
    st.caption("Capture a fruit image directly from your device camera.")
    camera_image = st.camera_input("Take a picture")
    if camera_image is None:
        st.info("Allow camera access and capture an image to continue.")
        return
    image = Image.open(camera_image).convert("RGB")
    left, right = st.columns(2)
    with left:
        st.image(image, caption="Captured image", use_container_width=True)
    with right:
        try:
            results = predict_image(model, load_and_preprocess_image(camera_image, IMAGE_SIZE), classes)
            show_prediction(results)
            save_prediction_history("camera_capture.jpg", results["predicted_class"], results["confidence_percentage"], "camera")
        except Exception as exc:
            st.error(f"Unable to classify the captured image: {exc}")


def history_page() -> None:
    st.header("📊 Prediction Dashboard")
    if not HISTORY_PATH.exists():
        st.info("No predictions have been recorded yet.")
        return
    df = pd.read_csv(HISTORY_PATH)
    if df.empty:
        st.info("No prediction records available.")
        return
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Predictions", len(df))
    col2.metric("Most predicted", str(df["Predicted Class"].mode().iloc[0]))
    col3.metric("Average confidence", f"{df['Confidence (%)'].mean():.1f}%")
    col4.metric("High confidence", f"{(df['Confidence (%)'] >= 85).mean() * 100:.1f}%")
    st.divider()
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download prediction history", df.to_csv(index=False).encode("utf-8"), file_name="fruit_prediction_history.csv", mime="text/csv")
    if st.button("🗑️ Clear prediction history"):
        HISTORY_PATH.unlink(missing_ok=True)
        st.rerun()


def about_page(classes: list[str]) -> None:
    st.header("ℹ️ About FruitVision AI")
    st.markdown("**FruitVision AI** is a CNN-based fruit image classification application with confidence scoring, top-3 predictions, multi-image upload, camera prediction, analytics, and CSV export.")
    st.subheader("Supported classes")
    st.write(" • ".join(classes))
    st.info("For reliable results, use a clear image with good lighting and a visible fruit.")


def main() -> None:
    st.markdown("<style>.block-container {padding-top: 2rem; padding-bottom: 3rem;} [data-testid=\"stMetricValue\"] {font-size: 1.6rem;}</style>", unsafe_allow_html=True)
    st.title("🍎 FruitVision AI")
    st.caption("Advanced Fruit Image Classification using Convolutional Neural Networks")
    try:
        model = load_model()
        classes = load_classes()
    except Exception as exc:
        st.error(str(exc))
        st.stop()
    with st.sidebar:
        st.title("Navigation")
        page = st.radio("Choose a mode", ["📸 Image Classifier", "📷 Camera", "📊 Dashboard", "ℹ️ About"])
        st.divider()
        st.success(f"Model loaded • {len(classes)} classes")
        st.caption("CNN inference engine: TensorFlow/Keras")
    if page == "📸 Image Classifier":
        upload_page(model, classes)
    elif page == "📷 Camera":
        camera_page(model, classes)
    elif page == "📊 Dashboard":
        history_page()
    else:
        about_page(classes)


if __name__ == "__main__":
    main()
