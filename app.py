"""Deployable Streamlit fruit image classifier with model-compatible preprocessing."""
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
    try:
        data = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
        if isinstance(data, list) and len(data) == 10 and all(isinstance(x, str) for x in data):
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return DEFAULT_CLASSES.copy()


@st.cache_resource(show_spinner="Loading fruit classifier...")
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


def find_model() -> Path | None:
    for path in MODEL_CANDIDATES:
        if path.exists():
            return path
    return None


def save_uploaded_model(data: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as f:
        f.write(data)
        return f.name


def get_model_image_size(model: tf.keras.Model) -> tuple[int, int]:
    """Read the required HxW from the actual loaded model.

    This prevents an old 128x128 model from being fed 160x160 images and
    also lets a newly trained 160x160 model run without changing the app.
    """
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
    image_size = get_model_image_size(model)
    arr = preprocess(image, image_size)
    h, w = arr.shape[:2]
    crop = arr[int(h * .08):int(h * .92), int(w * .08):int(w * .92)]
    crop_image = Image.fromarray(np.clip(crop * 255, 0, 255).astype(np.uint8))
    views = [
        arr,
        np.ascontiguousarray(arr[:, ::-1, :]),
        preprocess(crop_image, image_size),
    ]

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
    ood_threshold = None
    ood_pass = True
    nearest_classes = []
    if ood is not None:
        centroids, thresholds, global_threshold = ood
        embedding_model = get_embedding_model(model)
        embeddings = np.asarray(embedding_model.predict(np.stack(views), verbose=0), dtype=np.float32)
        if embeddings.shape[1] != centroids.shape[1]:
            ood = None
        else:
            distances = []
            for emb in embeddings:
                idx, dist = nearest_centroid_distance(emb, centroids)
                nearest_classes.append(idx)
                distances.append(dist)
            ood_distance = float(np.mean(distances))
            if thresholds is not None and len(thresholds) == len(classes):
                ood_threshold = float(thresholds[best])
            else:
                ood_threshold = global_threshold
            nearest_agreement = sum(x == best for x in nearest_classes) >= 2
            ood_pass = ood_distance <= ood_threshold and nearest_agreement

    supported = confidence >= ACCEPTANCE_THRESHOLD and margin >= MIN_MARGIN and agreement >= MIN_VIEW_AGREEMENT and ood_pass
    return {
        "class": classes[best], "confidence": confidence, "margin": margin,
        "agreement": agreement, "ood_distance": ood_distance, "ood_threshold": ood_threshold,
        "ood_pass": ood_pass, "supported": supported,
        "model_image_size": image_size,
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
    c4.metric("OOD check", "Passed" if result["ood_pass"] else "Rejected")
    st.caption(f"Model input: {result['model_image_size'][0]}×{result['model_image_size'][1]} pixels")
    st.subheader("🏆 Ranked predictions")
    for rank, (name, prob) in enumerate(result["top"], 1):
        st.write(f"**{rank}. {name}** — {prob * 100:.2f}%")
        st.progress(float(prob))
    if iq["warnings"]:
        st.warning(" • ".join(iq["warnings"]))
    if result["ood_distance"] is not None:
        st.caption(f"Feature-space unknown-image check: distance {result['ood_distance']:.3f} / class threshold {result['ood_threshold']:.3f}.")


# The remainder of the original dashboard/model/about UI follows unchanged.
# It is intentionally kept in this file so existing Streamlit navigation remains available.

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
    if bundled:
        model = load_model(str(bundled))
        st.success(f"✅ Bundled model: `{bundled.relative_to(BASE_DIR)}`")
        st.info(f"Model input size: {get_model_image_size(model)[0]}×{get_model_image_size(model)[1]} | Classes: {len(classes)}")
        with st.expander("View model architecture"):
            rows = [{"Layer": layer.name, "Type": layer.__class__.__name__} for layer in model.layers]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.error("No bundled model found.")


def main() -> None:
    classes = load_classes()
    bundled = find_model()
    page = st.sidebar.radio("Navigation", ["Classifier", "Camera", "Dashboard", "Model Setup", "About"])
    if page == "Dashboard":
        dashboard_page(classes); return
    if page == "Model Setup":
        model_setup_page(classes, bundled); return
    if page == "About":
        st.title("🍓 Fruit Image Classifier")
        st.write("Classifies Apple, Banana, Blackberry, Grape, Kiwi, Mango, Orange, Pineapple, Strawberry and Watermelon using a trained image-classification model.")
        return

    st.title("🍓 Fruit Image Classifier")
    if bundled is None:
        st.error("No trained model is available. Please add a .keras model to the models folder.")
        return
    model = load_model(str(bundled))
    if len(classes) != int(model.output_shape[-1]):
        st.error(f"Model/class metadata mismatch: model has {model.output_shape[-1]} outputs, metadata has {len(classes)} classes.")
        return

    image = None
    source = "Upload"
    if page == "Classifier":
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
        iq = quality(image)
        with st.spinner("Classifying image..."):
            result = predict(model, image, classes)
        show_result(result, iq)
        record_prediction(result, iq, source)


if __name__ == "__main__":
    main()
