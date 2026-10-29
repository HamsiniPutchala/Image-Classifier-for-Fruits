"""Deployable Streamlit fruit classifier with the original interactive layout."""
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
DEFAULT_IMAGE_SIZE = (160, 160)
ACCEPTANCE_THRESHOLD = 0.55
MIN_MARGIN = 0.08
MIN_VIEW_AGREEMENT = 2 / 3

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
    return tf.keras.models.load_model(path, compile=False)


@st.cache_resource(show_spinner="Loading unknown-image detector...")
def load_ood(path: str) -> tuple[np.ndarray, np.ndarray | None, float] | None:
    file = Path(path)
    if not file.exists():
        return None
    data = np.load(file)
    centroids = np.asarray(data["centroids"], dtype=np.float32)
    thresholds = np.asarray(data["distance_thresholds"], dtype=np.float32) if "distance_thresholds" in data else None
    global_threshold = float(data["distance_threshold"])
    return centroids, thresholds, global_threshold


def save_uploaded_model(data: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as f:
        f.write(data)
        return f.name


def find_model() -> Path | None:
    return next((p for p in MODEL_CANDIDATES if p.exists()), None)


def get_model_image_size(model: tf.keras.Model) -> tuple[int, int]:
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    if len(shape) != 4 or shape[1] is None or shape[2] is None:
        return DEFAULT_IMAGE_SIZE
    return int(shape[1]), int(shape[2])


def preprocess(image: Image.Image, image_size: tuple[int, int]) -> np.ndarray:
    image = image.convert("RGB").resize(image_size, Image.Resampling.LANCZOS)
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
    image_size = get_model_image_size(model)
    arr = preprocess(image, image_size)
    h, w = arr.shape[:2]
    crop = arr[int(h * .08):int(h * .92), int(w * .08):int(w * .92)]
    crop_image = Image.fromarray(np.clip(crop * 255, 0, 255).astype(np.uint8))
    views = [arr, np.ascontiguousarray(arr[:, ::-1, :]), preprocess(crop_image, image_size)]

    predictions = []
    for view in views:
        raw = np.asarray(model.predict(view[None, ...], verbose=0)[0], dtype=float)
        if len(raw) != len(classes):
            raise ValueError(f"Model has {len(raw)} outputs but class metadata has {len(classes)} classes.")
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

    ood_distance = None
    ood_threshold = None
    ood_pass = True
    ood = load_ood(str(OOD_PATH))
    if ood is not None:
        centroids, thresholds, global_threshold = ood
        embedding_model = get_embedding_model(model)
        embeddings = np.asarray(embedding_model.predict(np.stack(views), verbose=0), dtype=np.float32)
        if embeddings.shape[1] == centroids.shape[1]:
            distances = []
            nearest_classes = []
            for emb in embeddings:
                idx, dist = nearest_centroid_distance(emb, centroids)
                nearest_classes.append(idx)
                distances.append(dist)
            ood_distance = float(np.mean(distances))
            ood_threshold = float(thresholds[best]) if thresholds is not None and len(thresholds) == len(classes) else global_threshold
            nearest_agreement = sum(x == best for x in nearest_classes) >= 2
            ood_pass = ood_distance <= ood_threshold and nearest_agreement

    supported = confidence >= ACCEPTANCE_THRESHOLD and margin >= MIN_MARGIN and agreement >= MIN_VIEW_AGREEMENT and ood_pass
    return {
        "class": classes[best], "confidence": confidence, "margin": margin, "agreement": agreement,
        "ood_distance": ood_distance, "ood_threshold": ood_threshold, "ood_pass": ood_pass,
        "supported": supported, "model_image_size": image_size,
        "top": [(classes[int(i)], float(probs[int(i)])) for i in order[:5]],
    }


def record_prediction(result: dict[str, Any], image_quality: dict[str, Any], source: str) -> None:
    if "prediction_history" not in st.session_state:
        st.session_state["prediction_history"] = []
    st.session_state["prediction_history"].append({
        "Fruit": result["class"], "Confidence": result["confidence"], "Margin": result["margin"],
        "Agreement": result["agreement"], "OOD distance": result["ood_distance"],
        "Status": "Accepted" if result["supported"] else "Rejected", "Quality": image_quality["quality"], "Source": source,
    })


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
    st.caption(f"Model input: {result['model_image_size'][0]} × {result['model_image_size'][1]} pixels")
    for rank, (name, prob) in enumerate(result["top"], 1):
        st.write(f"**{rank}. {name}** — {prob * 100:.2f}%")
        st.progress(float(prob))
    if image_quality["warnings"]:
        st.warning(" • ".join(image_quality["warnings"]))
    if result["ood_distance"] is not None:
        st.caption(f"Feature-space check: distance {result['ood_distance']:.3f} / threshold {result['ood_threshold']:.3f}.")


def dashboard_page(classes: list[str]) -> None:
    st.title("📊 Prediction Dashboard")
    history = st.session_state.get("prediction_history", [])
    if not history:
        st.info("No predictions yet. Use **Classifier** or **Camera** to generate dashboard data.")
        a, b, c, d = st.columns(4)
        a.metric("Predictions", 0); b.metric("Accepted", 0); c.metric("Rejected", 0); d.metric("Avg. confidence", "—")
        return
    df = pd.DataFrame(history)
    accepted = int((df["Status"] == "Accepted").sum())
    rejected = int((df["Status"] == "Rejected").sum())
    avg_conf = float(df["Confidence"].mean())
    best_conf = float(df["Confidence"].max())
    a, b, c, d = st.columns(4)
    a.metric("🔎 Predictions", len(df), border=True); b.metric("✅ Accepted", accepted, border=True)
    c.metric("🚫 Rejected", rejected, border=True); d.metric("🎯 Avg. confidence", f"{avg_conf * 100:.1f}%", border=True)
    tab1, tab2, tab3 = st.tabs(["📈 Analytics", "🍓 Fruit Insights", "🧾 Prediction Log"])
    with tab1:
        x, y = st.columns(2)
        with x:
            st.subheader("Confidence by prediction"); st.line_chart(df[["Confidence"]], height=280)
        with y:
            st.subheader("Accepted vs rejected"); st.bar_chart(pd.DataFrame({"Count": [accepted, rejected]}, index=["Accepted", "Rejected"]), height=280)
        st.subheader("Prediction quality"); st.bar_chart(df.groupby("Quality").size().rename("Count"), height=220)
    with tab2:
        counts = df["Fruit"].value_counts().reindex(classes).fillna(0).astype(int)
        st.subheader("Most predicted fruits"); st.bar_chart(counts, height=350); st.caption(f"Highest recorded confidence: {best_conf * 100:.1f}%")
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
            st.write("The application is using the repository's trained MobileNetV2 model.")
            with st.expander("View model architecture", expanded=False):
                rows = [{"Layer": layer.name, "Type": layer.__class__.__name__} for layer in model.layers]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.warning("No trained model is bundled in the repository. Upload a `.keras` model for this session.")
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
        c1, c2 = st.columns(2)
        if bundled:
            model = load_model(str(bundled)); mh, mw = get_model_image_size(model)
            c1.metric("Input resolution", f"{mh} × {mw}")
        else:
            c1.metric("Input resolution", "Model dependent")
        c2.metric("Output classes", len(classes))
        st.write("**Architecture:** MobileNetV2 transfer learning")
        st.write("**Framework:** TensorFlow / Keras")
        st.write("**Prediction views:** Original + mirrored + center crop")
        st.write("**Acceptance threshold:** 55% confidence")
        st.write("**Minimum margin:** 8%")
        st.write("**Required view agreement:** 2 of 3")
    with tab3:
        checks = {"Class metadata": CLASS_NAMES_PATH.exists(), "Best model": (BASE_DIR / "models" / "best_model.keras").exists(), "Classifier model": (BASE_DIR / "models" / "fruit_classifier.keras").exists(), "10 classes configured": len(classes) == 10, "OOD calibration": OOD_PATH.exists()}
        for label, ok in checks.items():
            st.success(f"✓ {label}") if ok else st.error(f"✗ {label}")


def about_page(classes: list[str]) -> None:
    st.title("ℹ️ About the Project")
    tabs = st.tabs(["📖 Overview", "🧠 AI Model", "🔧 Technology Stack", "🛡️ Validation"])
    with tabs[0]:
        st.markdown("### 🍎 Image Classifier for Fruits")
        st.write("An interactive AI application that recognizes fruit images using a MobileNetV2 transfer-learning model and a Streamlit interface.")
        st.write("The application supports ten fruit categories and provides confidence-aware results for uploaded images and camera captures.")
        st.info("This project is designed as a deployable AI/ML internship project with an interactive user interface.")
    with tabs[1]:
        st.write("**Model:** MobileNetV2 pretrained backbone with a classification head")
        st.write("**Input:** Model-compatible RGB image")
        st.write("**Classes:** " + ", ".join(classes))
        st.write("**Inference:** Three-view probability averaging")
    with tabs[2]:
        tech = pd.DataFrame({"Technology": ["Python", "TensorFlow", "Keras", "Streamlit", "NumPy", "Pandas", "Pillow"], "Purpose": ["Application logic", "ML runtime", "Model loading", "Web interface", "Numerical processing", "Dashboard data", "Image processing"]})
        st.dataframe(tech, use_container_width=True, hide_index=True)
    with tabs[3]:
        st.write("The application combines confidence, top-two prediction margin, agreement across three image views and calibrated feature-space validation before accepting a prediction.")
        st.warning("The unsupported-input check is a heuristic for a closed-set classifier; it is not a guaranteed out-of-distribution detector.")


def home_page(classes: list[str], bundled: Path | None) -> None:
    st.markdown("""<style>.hero{padding:2.2rem 2.4rem;border-radius:24px;background:linear-gradient(135deg,#fff4e6,#fffaf4 48%,#eef9f1);border:1px solid rgba(0,0,0,.06);margin-bottom:1.4rem}.hero h1{font-size:3rem;margin:0 0 .55rem}.hero p{font-size:1.08rem;color:#4b5563;max-width:760px}.eyebrow{font-weight:700;letter-spacing:.08em;text-transform:uppercase;font-size:.78rem;color:#d97706}.card{padding:1.15rem;border:1px solid rgba(0,0,0,.08);border-radius:18px;background:rgba(255,255,255,.72);min-height:150px}.fruit-chip{display:inline-block;padding:.42rem .75rem;margin:.2rem .25rem .2rem 0;border-radius:999px;background:#f7f7f7;border:1px solid #e5e7eb;font-size:.88rem}</style>""", unsafe_allow_html=True)
    st.markdown("<div class='hero'><div class='eyebrow'>AI-Powered Fruit Recognition</div><h1>🍎 Fruit Image Classifier</h1><p>Identify fruits from images or camera captures using MobileNetV2 transfer learning with confidence and multi-view validation.</p></div>", unsafe_allow_html=True)
    m1,m2,m3,m4=st.columns(4,gap="medium")
    if bundled:
        model = load_model(str(bundled)); mh, mw = get_model_image_size(model); input_size = f"{mh} × {mw}"
    else:
        input_size = "Model dependent"
    m1.metric("🍓 Fruit classes",len(classes)); m2.metric("🖼️ Input size",input_size); m3.metric("🧠 Model","MobileNetV2"); m4.metric("🛡️ Views checked","3")
    st.subheader("✨ What you can do")
    f1,f2,f3=st.columns(3,gap="large")
    for col,title,text in [(f1,"📸 Upload & Predict","Upload JPG, JPEG or PNG images and receive ranked predictions."),(f2,"📷 Camera","Capture a fruit directly from your device camera."),(f3,"📊 Dashboard","Explore confidence, accepted/rejected results and prediction history.")]:
        with col: st.markdown(f"<div class='card'><h3>{title}</h3><p>{text}</p></div>",unsafe_allow_html=True)
    st.subheader("🍇 Supported fruits")
    st.markdown("".join(f"<span class='fruit-chip'>{x}</span>" for x in classes),unsafe_allow_html=True)
    if bundled: st.success("✅ Trained 10-class model is bundled and ready to use.")
    else: st.info("ℹ️ No bundled model was found. Open **Model Setup** to upload a trained `.keras` model for this session.")


def main() -> None:
    classes = load_classes()
    bundled = find_model()
    st.sidebar.title("🍎 Fruit Classifier")
    st.sidebar.caption("MobileNetV2 • TensorFlow/Keras • Streamlit")
    page = st.sidebar.radio("Navigate", ["🏠 Home", "📸 Classifier", "📷 Camera", "📊 Dashboard", "⚙️ Model Setup", "ℹ️ About"])

    if page == "🏠 Home":
        home_page(classes, bundled); return
    if page == "📊 Dashboard":
        dashboard_page(classes); return
    if page == "⚙️ Model Setup":
        model_setup_page(classes, bundled); return
    if page == "ℹ️ About":
        about_page(classes); return

    st.title("📸 Classifier" if page == "📸 Classifier" else "📷 Camera")
    if bundled is None:
        st.error("No trained model is available. Please add a .keras model to the models folder or use Model Setup.")
        return
    model = load_model(str(bundled))
    if len(classes) != int(model.output_shape[-1]):
        st.error(f"Model/class metadata mismatch: model has {model.output_shape[-1]} outputs, metadata has {len(classes)} classes.")
        return

    image = None
    source = "Upload"
    if page == "📸 Classifier":
        uploaded = st.file_uploader("Upload a fruit image", type=["jpg", "jpeg", "png", "webp"])
        if uploaded is not None:
            image = Image.open(uploaded)
    else:
        source = "Camera"
        camera = st.camera_input("Take a fruit photo")
        if camera is not None:
            image = Image.open(camera)

    if image is not None:
        st.image(image, caption="Input image", width=350)
        image_quality = quality(image)
        with st.spinner("Classifying image..."):
            result = predict(model, image, classes)
        show_result(result, image_quality)
        record_prediction(result, image_quality, source)


if __name__ == "__main__":
    main()
