"""Deployable Streamlit fruit image classifier with calibrated unknown-image rejection."""
from __future__ import annotations

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
OOD_PATH = BASE_DIR / "models" / "ood_calibration.npz"
DEFAULT_CLASSES = ["Apple", "Banana", "Blackberry", "Grape", "Kiwi", "Mango", "Orange", "Pineapple", "Strawberry", "Watermelon"]
IMAGE_SIZE = (128, 128)
ACCEPTANCE_THRESHOLD = 0.65
MIN_MARGIN = 0.10
MIN_VIEW_AGREEMENT = 2 / 3

st.set_page_config(page_title="Fruit Image Classifier", page_icon="🍎", layout="wide")


def load_classes() -> list[str]:
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


@st.cache_resource(show_spinner="Loading unknown-image detector...")
def load_ood(path: str) -> tuple[np.ndarray, float] | None:
    file = Path(path)
    if not file.exists():
        return None
    data = np.load(file)
    return np.asarray(data["centroids"], dtype=np.float32), float(data["distance_threshold"])


def find_model() -> Path | None:
    for path in MODEL_CANDIDATES:
        if path.exists():
            return path
    return None


def save_uploaded_model(data: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as f:
        f.write(data)
        return f.name


def preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32) / 255.0


def quality(image: Image.Image) -> dict[str, Any]:
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    warnings = []
    if float(gray.mean()) < 45:
        warnings.append("Image may be too dark")
    if float(gray.mean()) > 215:
        warnings.append("Image may be overexposed")
    if float(gray.std()) < 25:
        warnings.append("Low contrast")
    return {"quality": "Good" if not warnings else "Needs improvement", "warnings": warnings}


def get_embedding_model(model: tf.keras.Model) -> tf.keras.Model:
    try:
        layer = model.get_layer("global_average_pooling")
    except ValueError:
        return tf.keras.Model(model.input, model.layers[-3].output)
    return tf.keras.Model(model.input, layer.output)


def nearest_centroid_distance(embedding: np.ndarray, centroids: np.ndarray) -> tuple[int, float]:
    distances = np.linalg.norm(centroids - embedding[None, :], axis=1)
    index = int(np.argmin(distances))
    return index, float(distances[index])


def predict(model: tf.keras.Model, image: Image.Image, classes: list[str]) -> dict[str, Any]:
    arr = preprocess(image)
    h, w = arr.shape[:2]
    crop = arr[int(h * .08):int(h * .92), int(w * .08):int(w * .92)]
    views = [arr, np.ascontiguousarray(arr[:, ::-1, :]), preprocess(Image.fromarray(np.clip(crop * 255, 0, 255).astype(np.uint8)))]
    predictions = []
    for view in views:
        raw = np.asarray(model.predict(view[None, ...], verbose=0)[0], dtype=float)
        if len(raw) != len(classes):
            raise ValueError(f"Model has {len(raw)} outputs but metadata has {len(classes)} classes.")
        raw = np.clip(raw, 0, None)
        total = raw.sum()
        if total <= 0:
            raise ValueError("Model returned an invalid probability vector.")
        predictions.append(raw / total)

    probs = np.mean(predictions, axis=0)
    order = np.argsort(probs)[::-1]
    best = int(order[0])
    confidence = float(probs[best])
    margin = confidence - float(probs[int(order[1])])
    agreement = sum(int(np.argmax(p)) == best for p in predictions) / len(predictions)

    ood = load_ood(str(OOD_PATH))
    ood_distance = None
    ood_pass = True
    if ood is not None:
        centroids, threshold = ood
        embedding_model = get_embedding_model(model)
        embeddings = np.asarray(embedding_model.predict(np.stack(views), verbose=0), dtype=np.float32)
        nearest = []
        distances = []
        for emb in embeddings:
            idx, dist = nearest_centroid_distance(emb, centroids)
            nearest.append(idx)
            distances.append(dist)
        ood_distance = float(np.mean(distances))
        ood_pass = ood_distance <= threshold and sum(x == best for x in nearest) >= 2

    supported = confidence >= ACCEPTANCE_THRESHOLD and margin >= MIN_MARGIN and agreement >= MIN_VIEW_AGREEMENT and ood_pass
    return {
        "class": classes[best],
        "confidence": confidence,
        "margin": margin,
        "agreement": agreement,
        "ood_distance": ood_distance,
        "supported": supported,
        "top": [(classes[int(i)], float(probs[int(i)])) for i in order[:5]],
    }


def record_prediction(result: dict[str, Any], iq: dict[str, Any], source: str) -> None:
    history = st.session_state.setdefault("prediction_history", [])
    history.append({"Fruit": result["class"], "Confidence": result["confidence"], "Margin": result["margin"], "Agreement": result["agreement"], "OOD distance": result["ood_distance"], "Status": "Accepted" if result["supported"] else "Unsupported / review", "Quality": iq["quality"], "Source": source})


def show_result(result: dict[str, Any], iq: dict[str, Any]) -> None:
    if result["supported"]:
        st.success(f"🍓 Predicted fruit: **{result['class']}**")
    else:
        st.error("🚫 Unsupported or unclear image")
        st.write("This image did not pass the fruit-class validation checks. Please upload a clear image of one of the 10 supported fruits.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Confidence", f"{result['confidence'] * 100:.2f}%")
    c2.metric("View agreement", f"{result['agreement'] * 100:.0f}%")
    c3.metric("Image quality", iq["quality"])
    c4.metric("OOD check", "Passed" if result["ood_distance"] is None or result["supported"] else "Rejected")
    st.subheader("🏆 Ranked predictions")
    for rank, (name, prob) in enumerate(result["top"], 1):
        st.write(f"**{rank}. {name}** — {prob * 100:.2f}%")
        st.progress(float(prob))
    if iq["warnings"]:
        st.warning(" • ".join(iq["warnings"]))
    if result["ood_distance"] is not None:
        st.caption(f"Feature-space unknown-image check: distance {result['ood_distance']:.3f} (calibrated from validation fruit images).")


def dashboard_page(classes: list[str]) -> None:
    st.title("📊 Prediction Dashboard")
    history = st.session_state.get("prediction_history", [])
    if not history:
        st.info("No predictions yet. Use Classifier or Camera to generate dashboard data.")
        a, b, c, d = st.columns(4)
        a.metric("Predictions", 0); b.metric("Accepted", 0); c.metric("Unsupported/review", 0); d.metric("Avg. confidence", "—")
        return
    df = pd.DataFrame(history)
    accepted = int((df["Status"] == "Accepted").sum())
    review = len(df) - accepted
    a, b, c, d = st.columns(4)
    a.metric("🔎 Predictions", len(df)); b.metric("✅ Accepted", accepted); c.metric("🚫 Unsupported/review", review); d.metric("🎯 Avg. confidence", f"{df['Confidence'].mean() * 100:.1f}%")
    tab1, tab2, tab3 = st.tabs(["📈 Analytics", "🍓 Fruit Insights", "🧾 Prediction Log"])
    with tab1:
        x, y = st.columns(2)
        with x:
            st.subheader("Confidence by prediction"); st.line_chart(df[["Confidence"]], height=280)
        with y:
            st.subheader("Accepted vs rejected"); st.bar_chart(pd.DataFrame({"Count": [accepted, review]}, index=["Accepted", "Rejected"]), height=280)
        st.subheader("Image quality"); st.bar_chart(df.groupby("Quality").size().rename("Count"), height=220)
    with tab2:
        counts = df["Fruit"].value_counts().reindex(classes).fillna(0).astype(int)
        st.subheader("Most predicted fruits"); st.bar_chart(counts, height=350)
    with tab3:
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download prediction history", df.to_csv(index=False).encode("utf-8"), "fruit_prediction_history.csv", "text/csv")
        if st.button("🗑️ Clear dashboard history"):
            st.session_state["prediction_history"] = []
            st.rerun()


def model_setup_page(classes: list[str], bundled: Path | None) -> None:
    st.title("⚙️ Interactive Model Setup")
    tab1, tab2, tab3 = st.tabs(["🤖 Model", "📋 Configuration", "🧪 Diagnostics"])
    with tab1:
        if bundled:
            st.success(f"✅ Bundled model: `{bundled.relative_to(BASE_DIR)}`")
            model = load_model(str(bundled))
            with st.expander("View model architecture"):
                rows = [{"Layer": layer.name, "Type": layer.__class__.__name__} for layer in model.layers]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.warning("No trained model is bundled. Upload a `.keras` model for this session.")
            uploaded = st.file_uploader("Upload trained model", type=["keras"], key="model_upload")
            if uploaded:
                path = save_uploaded_model(uploaded.getvalue())
                try:
                    model = load_model(path)
                    outputs = int(model.output_shape[-1])
                    if outputs != len(classes):
                        st.error(f"Uploaded model has {outputs} outputs; this app requires {len(classes)}.")
                    else:
                        st.session_state["uploaded_model_path"] = path
                        st.success("✅ Compatible model loaded successfully.")
                except Exception as exc:
                    st.error(f"Could not load model: {exc}")
    with tab2:
        a, b = st.columns(2)
        a.metric("Input resolution", "128 × 128"); b.metric("Output classes", len(classes))
        st.write("**Architecture:** MobileNetV2 transfer learning + fine-tuning")
        st.write("**Framework:** TensorFlow / Keras")
        st.write("**Prediction views:** Original + mirrored + center crop")
        st.write(f"**Confidence threshold:** {ACCEPTANCE_THRESHOLD * 100:.0f}%")
        st.write(f"**Minimum margin:** {MIN_MARGIN * 100:.0f}%")
        st.write(f"**Required view agreement:** {MIN_VIEW_AGREEMENT * 100:.0f}%")
        st.write(f"**Unknown-image calibration:** {'Available' if OOD_PATH.exists() else 'Not available'}")
    with tab3:
        checks = {"Class metadata": CLASS_NAMES_PATH.exists(), "Best model": (BASE_DIR / "models" / "best_model.keras").exists(), "Classifier model": (BASE_DIR / "models" / "fruit_classifier.keras").exists(), "OOD calibration": OOD_PATH.exists(), "10 classes configured": len(classes) == 10}
        for label, ok in checks.items():
            if ok:
                st.success(f"✓ {label}")
            else:
                st.error(f"✗ {label}")


def about_page(classes: list[str]) -> None:
    st.title("ℹ️ About the Project")
    tabs = st.tabs(["📖 Overview", "🧠 AI Model", "🔧 Technology Stack", "🛡️ Validation"])
    with tabs[0]:
        st.subheader("🍎 Image Classifier for Fruits")
        st.write("An interactive AI application that recognizes ten supported fruit categories using MobileNetV2 transfer learning, fine-tuning, multi-view prediction and a calibrated unknown-image detector.")
        st.write("Images outside the learned fruit feature space are rejected instead of being presented as a confident supported fruit.")
    with tabs[1]:
        st.write("**Model:** MobileNetV2 ImageNet backbone + fine-tuned classification head")
        st.write("**Input:** 128 × 128 RGB image")
        st.write("**Classes:** " + ", ".join(classes))
        st.write("**Inference:** Three-view probability averaging + feature-space OOD check")
    with tabs[2]:
        tech = pd.DataFrame({"Technology": ["Python", "TensorFlow", "Keras", "Streamlit", "NumPy", "Pandas", "Pillow"], "Purpose": ["Application logic", "ML runtime", "Model training/inference", "Web interface", "Numerical processing", "Dashboard data", "Image processing"]})
        st.dataframe(tech, use_container_width=True, hide_index=True)
    with tabs[3]:
        st.write("A prediction is accepted only when confidence, top-two margin, multi-view agreement and calibrated feature distance pass their checks.")
        st.warning("Unknown-image rejection is calibrated from validation fruit images; it improves closed-set safety but cannot guarantee perfect rejection for every possible non-fruit image.")


def home_page(classes: list[str], bundled: Path | None) -> None:
    st.markdown("""<style>.hero{padding:2.2rem 2.4rem;border-radius:24px;background:linear-gradient(135deg,#fff4e6,#fffaf4 48%,#eef9f1);border:1px solid rgba(0,0,0,.06);margin-bottom:1.4rem}.hero h1{font-size:3rem;margin:0 0 .55rem}.hero p{font-size:1.08rem;color:#4b5563;max-width:760px}.eyebrow{font-weight:700;letter-spacing:.08em;text-transform:uppercase;font-size:.78rem;color:#d97706}.card{padding:1.15rem;border:1px solid rgba(0,0,0,.08);border-radius:18px;background:rgba(255,255,255,.72);min-height:150px}.fruit-chip{display:inline-block;padding:.42rem .75rem;margin:.2rem .25rem .2rem 0;border-radius:999px;background:#f7f7f7;border:1px solid #e5e7eb;font-size:.88rem}</style>""", unsafe_allow_html=True)
    st.markdown("<div class='hero'><div class='eyebrow'>AI-Powered Fruit Recognition</div><h1>🍎 Fruit Image Classifier</h1><p>Identify supported fruits while filtering unclear or out-of-class images using fine-tuned MobileNetV2 and feature-space validation.</p></div>", unsafe_allow_html=True)
    a, b, c, d = st.columns(4)
    a.metric("🍓 Fruit classes", len(classes)); b.metric("🖼️ Input size", "128 × 128"); c.metric("🧠 Model", "MobileNetV2"); d.metric("🛡️ OOD filter", "Enabled" if OOD_PATH.exists() else "Pending")
    st.subheader("✨ What you can do")
    f1, f2, f3 = st.columns(3)
    cards = [(f1, "📸 Upload & Predict", "Upload images and receive validated ranked predictions."), (f2, "📷 Camera", "Capture a fruit directly from your device camera."), (f3, "📊 Dashboard", "Explore confidence, validation status and prediction history.")]
    for col, title, text in cards:
        with col:
            st.markdown(f"<div class='card'><h3>{title}</h3><p>{text}</p></div>", unsafe_allow_html=True)
    st.subheader("🍇 Supported fruits")
    st.markdown("".join(f"<span class='fruit-chip'>{x}</span>" for x in classes), unsafe_allow_html=True)
    if bundled:
        st.success("✅ Trained model is ready for inference.")


def classifier_page(model: tf.keras.Model, classes: list[str]) -> None:
    st.title("📸 Fruit Classifier")
    uploaded = st.file_uploader("Upload a JPG, JPEG or PNG image", type=["jpg", "jpeg", "png"], key="fruit_upload")
    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Uploaded image", use_container_width=True)
        iq = quality(image)
        try:
            result = predict(model, image, classes)
            show_result(result, iq)
            record_prediction(result, iq, "Upload")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


def camera_page(model: tf.keras.Model, classes: list[str]) -> None:
    st.title("📷 Camera Classifier")
    captured = st.camera_input("Capture a fruit image")
    if captured:
        image = Image.open(captured)
        st.image(image, caption="Captured image", use_container_width=True)
        iq = quality(image)
        try:
            result = predict(model, image, classes)
            show_result(result, iq)
            record_prediction(result, iq, "Camera")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


def main() -> None:
    classes = load_classes()
    bundled = find_model()
    st.sidebar.title("🍎 Fruit AI")
    page = st.sidebar.radio("Navigate", ["🏠 Home", "📸 Classifier", "📷 Camera", "📊 Dashboard", "⚙️ Model Setup", "ℹ️ About"])
    if page == "🏠 Home":
        home_page(classes, bundled); return
    if page == "📊 Dashboard":
        dashboard_page(classes); return
    if page == "⚙️ Model Setup":
        model_setup_page(classes, bundled); return
    if page == "ℹ️ About":
        about_page(classes); return
    if bundled is None:
        st.error("No trained model found. Run the training workflow or upload a model in Model Setup.")
        return
    model = load_model(str(bundled))
    if page == "📸 Classifier":
        classifier_page(model, classes)
    else:
        camera_page(model, classes)


if __name__ == "__main__":
    main()
