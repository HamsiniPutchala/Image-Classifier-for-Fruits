"""Image Classifier for Fruits - interactive Streamlit fruit classifier."""
from __future__ import annotations

import csv
import io
import json
from datetime import datetime
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
DATASET_PATH = BASE_DIR / "dataset"
UPLOADS_PATH = BASE_DIR / "uploads"
HISTORY_PATH = UPLOADS_PATH / "prediction_history.csv"
DEFAULT_CLASSES = ["Apple", "Banana", "Orange"]
IMAGE_SIZE = (128, 128)
UNKNOWN_THRESHOLD = 0.85
UPLOADS_PATH.mkdir(exist_ok=True)

st.set_page_config(
    page_title="Image Classifier for Fruits",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------------------- Styling -----------------------------
def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1450px;}
        [data-testid="stMetric"] {border: 1px solid rgba(120,120,120,.18); padding: 14px; border-radius: 16px; background: rgba(255,255,255,.035);}
        [data-testid="stMetricValue"] {font-size: 1.65rem;}
        .hero {padding: 28px 30px; border-radius: 24px; margin-bottom: 22px; background: linear-gradient(135deg, rgba(255,165,0,.20), rgba(255,80,120,.12), rgba(80,130,255,.14)); border: 1px solid rgba(255,255,255,.12);}
        .hero h1 {font-size: 2.55rem; margin: 0 0 8px 0;}
        .hero p {font-size: 1.05rem; opacity: .82; margin: 0;}
        .card {padding: 18px; border-radius: 18px; border: 1px solid rgba(120,120,120,.18); background: rgba(255,255,255,.035); height: 100%;}
        .fruit-card {text-align:center; padding: 18px 10px; border-radius: 18px; border: 1px solid rgba(120,120,120,.18); background: rgba(255,255,255,.035);}
        .fruit-icon {font-size: 2.5rem;}
        .small-muted {opacity: .68; font-size: .88rem;}
        .result-title {font-size: 1.55rem; font-weight: 700; margin-bottom: 4px;}
        .pill {display:inline-block; padding:5px 11px; border-radius:999px; background:rgba(46,204,113,.14); border:1px solid rgba(46,204,113,.30); font-size:.84rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Loading trained MobileNetV2 model...")
def load_model_from_path(path: str) -> tf.keras.Model:
    return tf.keras.models.load_model(path)


def find_bundled_model() -> Path | None:
    return next((path for path in MODEL_CANDIDATES if path.exists()), None)


@st.cache_data
def load_classes() -> list[str]:
    if CLASS_NAMES_PATH.exists():
        try:
            with CLASS_NAMES_PATH.open("r", encoding="utf-8") as file:
                classes = json.load(file)
            if isinstance(classes, list) and all(isinstance(x, str) for x in classes):
                return classes
        except (OSError, json.JSONDecodeError):
            pass
    train_path = DATASET_PATH / "train"
    if train_path.exists():
        classes = sorted(p.name for p in train_path.iterdir() if p.is_dir())
        if classes:
            return classes
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
    gray = np.asarray(image.convert("L"), dtype=np.float32)
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
    return {
        "quality": "Good" if not warnings else "Needs improvement",
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": sharpness,
        "warnings": warnings,
    }


def predict_image(model: tf.keras.Model, image_array: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    raw = np.asarray(model.predict(np.expand_dims(image_array, axis=0), verbose=0)[0], dtype=float)
    if len(raw) == 0:
        raise ValueError("The model returned no predictions.")
    if np.any(raw < 0) or not np.isclose(raw.sum(), 1.0, atol=0.05):
        exp = np.exp(raw - np.max(raw))
        probabilities = exp / exp.sum()
    else:
        probabilities = np.clip(raw, 0, None)
        probabilities /= probabilities.sum() if probabilities.sum() else 1
    if len(probabilities) != len(class_names):
        raise ValueError(f"Model has {len(probabilities)} outputs but class metadata contains {len(class_names)} classes.")
    order = np.argsort(probabilities)[::-1]
    top = [
        {
            "class": class_names[int(i)],
            "confidence": float(probabilities[int(i)]),
            "percentage": float(probabilities[int(i)] * 100),
        }
        for i in order[: min(3, len(order))]
    ]
    best = top[0]
    return {
        "predicted_class": best["class"],
        "confidence": best["confidence"],
        "confidence_percentage": best["percentage"],
        "is_supported": best["confidence"] >= UNKNOWN_THRESHOLD,
        "top_predictions": top,
        "probabilities": {class_names[int(i)]: float(probabilities[int(i)]) for i in order},
    }


def save_prediction_history(image_name: str, predicted_class: str, confidence: float, method: str, quality: str) -> None:
    exists = HISTORY_PATH.exists()
    with HISTORY_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not exists:
            writer.writerow(["Timestamp", "Image Name", "Predicted Class", "Confidence (%)", "Method", "Image Quality"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            image_name,
            predicted_class,
            f"{confidence:.2f}",
            method,
            quality,
        ])


def show_prediction(results: dict[str, Any], quality: dict[str, Any]) -> None:
    confidence = results["confidence"]
    if not results["is_supported"]:
        st.error("🚫 No supported fruit detected")
        st.info("This classifier currently recognizes only Apple, Banana, and Orange. Try a clear image containing one of these fruits.")
        return

    st.success(f"🍓 Predicted fruit: **{results['predicted_class']}**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Confidence", f"{results['confidence_percentage']:.2f}%")
    c2.metric("Assessment", confidence_label(confidence))
    c3.metric("Image quality", quality["quality"])
    st.progress(confidence, text=f"Top prediction confidence: {results['confidence_percentage']:.1f}%")

    if quality["warnings"]:
        st.warning(" • ".join(quality["warnings"]))

    st.markdown("#### 🏆 Top predictions")
    for rank, item in enumerate(results["top_predictions"], start=1):
        st.write(f"**{rank}. {item['class']}** — {item['percentage']:.2f}%")
        st.progress(item["confidence"])

    probability_df = pd.DataFrame({
        "Fruit": list(results["probabilities"].keys()),
        "Probability (%)": [v * 100 for v in results["probabilities"].values()],
    })
    st.markdown("#### 📈 Probability comparison")
    st.bar_chart(probability_df.set_index("Fruit"), height=280)


def classify_bytes(model: tf.keras.Model, data: bytes, classes: list[str]) -> tuple[Image.Image, dict[str, Any], dict[str, Any]]:
    image = Image.open(io.BytesIO(data)).convert("RGB")
    quality = image_quality(image)
    results = predict_image(model, preprocess_pil(image), classes)
    return image, results, quality


def home_page(classes: list[str], model_path: Path | None) -> None:
    st.markdown(
        "<div class='hero'><h1>🍎 Image Classifier for Fruits</h1><p>Explore an interactive fruit-recognition application powered by a trained MobileNetV2 transfer-learning model.</p></div>",
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Supported fruits", len(classes))
    c2.metric("Input size", "128 × 128")
    c3.metric("Model", "MobileNetV2")
    c4.metric("Mode", "Image + Camera")

    st.markdown("### 🍓 What can you do?")
    cards = [
        ("📸", "Classify Images", "Upload one or multiple fruit images and inspect confidence and ranked predictions."),
        ("📷", "Use Your Camera", "Capture a fruit directly from your device camera for instant inference."),
        ("📊", "Explore Analytics", "Review prediction history, confidence trends and downloadable results."),
    ]
    cols = st.columns(3)
    for col, (icon, title, text) in zip(cols, cards):
        with col:
            st.markdown(f"<div class='card'><div style='font-size:2rem'>{icon}</div><h4>{title}</h4><p class='small-muted'>{text}</p></div>", unsafe_allow_html=True)

    st.markdown("### 🍎 Supported classes")
    icons = {"Apple": "🍎", "Banana": "🍌", "Orange": "🍊"}
    cols = st.columns(len(classes))
    for col, fruit in zip(cols, classes):
        with col:
            st.markdown(f"<div class='fruit-card'><div class='fruit-icon'>{icons.get(fruit, '🍓')}</div><b>{fruit}</b><br><span class='small-muted'>Supported class</span></div>", unsafe_allow_html=True)

    st.divider()
    if model_path:
        st.success(f"✅ Trained model ready: **{model_path.name}**")
    st.caption("Tip: use a clear, well-lit image with the fruit visible and reasonably centered for more reliable predictions.")


def upload_page(model: tf.keras.Model, classes: list[str]) -> None:
    st.header("📸 Image Classification")
    st.caption("Upload JPG/PNG images for interactive classification, quality checks and ranked probabilities.")
    files = st.file_uploader("Choose fruit images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if not files:
        st.info("👆 Upload one or more images to start.")
        return
    all_rows = []
    for uploaded_file in files:
        st.divider()
        left, right = st.columns([0.9, 1.4])
        data = uploaded_file.getvalue()
        with left:
            image = Image.open(io.BytesIO(data)).convert("RGB")
            st.image(image, caption=uploaded_file.name, use_container_width=True)
            st.caption(f"Resolution: {image.width} × {image.height}px")
        with right:
            try:
                _, results, quality = classify_bytes(model, data, classes)
                show_prediction(results, quality)
                if results["is_supported"]:
                    save_prediction_history(uploaded_file.name, results["predicted_class"], results["confidence_percentage"], "upload", quality["quality"])
                    all_rows.append({
                        "Image": uploaded_file.name,
                        "Prediction": results["predicted_class"],
                        "Confidence (%)": round(results["confidence_percentage"], 2),
                        "Quality": quality["quality"],
                    })
            except Exception as exc:
                st.error(f"Could not classify {uploaded_file.name}: {exc}")
    if all_rows:
        st.subheader("📋 Batch results")
        batch_df = pd.DataFrame(all_rows)
        st.dataframe(batch_df, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download batch results", batch_df.to_csv(index=False).encode("utf-8"), "fruit_batch_results.csv", "text/csv")


def camera_page(model: tf.keras.Model, classes: list[str]) -> None:
    st.header("📷 Camera Classification")
    st.caption("Capture a fruit image and run the same trained model used by the image classifier.")
    camera_image = st.camera_input("Capture a fruit")
    if camera_image is None:
        st.info("📷 Allow camera access and capture an image when ready.")
        return
    try:
        image, results, quality = classify_bytes(model, camera_image.getvalue(), classes)
        st.image(image, caption="Captured image", width=450)
        show_prediction(results, quality)
        if results["is_supported"]:
            save_prediction_history("camera_capture.jpg", results["predicted_class"], results["confidence_percentage"], "camera", quality["quality"])
    except Exception as exc:
        st.error(f"Camera prediction failed: {exc}")


def dashboard_page() -> None:
    st.header("📊 Prediction Analytics")
    st.caption("A visual summary of successful predictions recorded by the application.")
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
    left, right = st.columns(2)
    with left:
        st.subheader("🍓 Prediction distribution")
        st.bar_chart(df["Predicted Class"].value_counts().rename("Predictions"), height=300)
    with right:
        st.subheader("🎯 Confidence by prediction")
        st.line_chart(df["Confidence (%)"], height=300)
    st.subheader("🧾 Prediction history")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Export complete history", df.to_csv(index=False).encode("utf-8"), "fruit_prediction_history.csv", "text/csv")
    if st.button("🗑️ Clear history"):
        HISTORY_PATH.unlink(missing_ok=True)
        st.rerun()


def model_page(model_path: Path | None, classes: list[str]) -> None:
    st.header("⚙️ Model Setup")
    st.caption("Technical information about the trained model bundled with this application.")

    if model_path is None:
        st.error("No trained model was found in the repository.")
        st.code("models/best_model.keras", language="text")
        st.info("The application expects the trained Keras model and class metadata inside the models/ directory.")
        return

    try:
        model = load_model_from_path(str(model_path))
        input_shape = model.input_shape
        output_shape = model.output_shape
    except Exception as exc:
        st.error(f"The bundled model exists but could not be loaded: {exc}")
        return

    st.success(f"🟢 Model loaded successfully — **{model_path.name}**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Architecture", "MobileNetV2")
    c2.metric("Input", "128 × 128 × 3")
    c3.metric("Outputs", len(classes))
    c4.metric("Reject threshold", "85%")

    st.markdown("### 🔧 Configuration")
    config_df = pd.DataFrame([
        ["Framework", "TensorFlow / Keras"],
        ["Model approach", "MobileNetV2 transfer learning"],
        ["Input image", "RGB, resized to 128 × 128"],
        ["Output classes", ", ".join(classes)],
        ["Prediction rule", "Top class accepted when confidence ≥ 85%"],
        ["Model file", str(model_path.relative_to(BASE_DIR))],
        ["Class metadata", "models/class_names.json"],
        ["Detected input shape", str(input_shape)],
        ["Detected output shape", str(output_shape)],
    ], columns=["Setting", "Value"])
    st.dataframe(config_df, use_container_width=True, hide_index=True)

    st.markdown("### 🧠 How inference works")
    steps = [
        ("1", "Image input", "The uploaded or captured image is converted to RGB."),
        ("2", "Preprocessing", "The image is resized to 128 × 128 and pixel values are scaled to 0–1."),
        ("3", "MobileNetV2 inference", "The trained Keras model produces class probabilities."),
        ("4", "Confidence check", "The highest probability is compared with the 85% supported-image threshold."),
        ("5", "Interactive result", "Supported predictions show confidence and ranked probabilities; low-confidence inputs are rejected."),
    ]
    for number, title, description in steps:
        st.markdown(f"**{number}. {title}** — {description}")

    with st.expander("View model layer summary"):
        rows = []
        for layer in model.layers:
            rows.append({"Layer": layer.name, "Type": layer.__class__.__name__, "Parameters": layer.count_params()})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def about_page(classes: list[str], model_path: Path | None) -> None:
    st.header("ℹ️ About Image Classifier for Fruits")
    st.markdown(
        "<div class='hero'><h1>🍎 A visual fruit-recognition experience</h1><p>A Streamlit application that uses a trained MobileNetV2 transfer-learning model to classify supported fruit images.</p></div>",
        unsafe_allow_html=True,
    )

    st.markdown("### 🎯 Project overview")
    st.write(
        "This project demonstrates an end-to-end image-classification workflow: a trained Keras model is packaged with a Streamlit interface, images are preprocessed consistently, predictions are scored with confidence, and successful results can be explored through history and analytics."
    )

    st.markdown("### 🧩 Technology stack")
    tech_cols = st.columns(5)
    for col, (icon, name) in zip(tech_cols, [("🧠", "MobileNetV2"), ("⚙️", "TensorFlow"), ("🔗", "Keras"), ("🖥️", "Streamlit"), ("🖼️", "Pillow")]):
        with col:
            st.markdown(f"<div class='fruit-card'><div class='fruit-icon'>{icon}</div><b>{name}</b></div>", unsafe_allow_html=True)

    st.markdown("### 🍓 Supported fruits")
    st.write(" • ".join(classes))
    st.info("Images outside these trained classes are not valid target classes. The app uses an 85% confidence threshold to reject low-confidence inputs rather than presenting them as reliable fruit predictions.")

    st.markdown("### 📐 Prediction pipeline")
    st.markdown("**Image → RGB conversion → 128×128 resize → normalization → MobileNetV2 model → class probabilities → confidence check → visual result**")

    st.markdown("### ✨ Interactive features")
    features = [
        "📸 Single or batch image upload",
        "📷 Camera-based classification",
        "🏆 Top prediction and ranked probabilities",
        "🎯 Confidence visualization",
        "🔍 Image-quality checks for brightness, contrast and sharpness",
        "📊 Prediction analytics and history",
        "⬇️ Downloadable CSV reports",
        "🚫 Low-confidence unsupported-input rejection",
    ]
    for feature in features:
        st.markdown(f"- {feature}")

    if model_path:
        st.caption(f"Bundled model: `{model_path.relative_to(BASE_DIR)}`")


def main() -> None:
    inject_styles()
    classes = load_classes()
    model_path = find_bundled_model()

    with st.sidebar:
        st.markdown("## 🍎 Fruit Classifier")
        st.caption("AI-powered image recognition")
        st.divider()
        page = st.radio(
            "Navigate",
            ["🏠 Home", "📸 Classifier", "📷 Camera", "📊 Dashboard", "⚙️ Model Setup", "ℹ️ About"],
            label_visibility="collapsed",
        )
        st.divider()
        if model_path:
            st.success("🟢 Model ready")
        else:
            st.error("🔴 Model missing")
        st.caption(f"{len(classes)} supported classes")
        st.caption("TensorFlow / Keras • Streamlit")

    if page == "🏠 Home":
        home_page(classes, model_path)
        return
    if page == "⚙️ Model Setup":
        model_page(model_path, classes)
        return
    if page == "ℹ️ About":
        about_page(classes, model_path)
        return

    if model_path is None:
        st.error("A trained model is required for classification. Open **Model Setup** for details.")
        return
    try:
        model = load_model_from_path(str(model_path))
    except Exception as exc:
        st.error(f"The trained model could not be loaded: {exc}")
        return

    if page == "📸 Classifier":
        upload_page(model, classes)
    elif page == "📷 Camera":
        camera_page(model, classes)
    elif page == "📊 Dashboard":
        dashboard_page()


if __name__ == "__main__":
    main()
