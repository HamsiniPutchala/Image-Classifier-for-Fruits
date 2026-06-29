"""Streamlit web application for fruit image classification."""

import os
import cv2
import csv
import logging
from datetime import datetime
from typing import Optional
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image
import pandas as pd

from utils import (
    load_and_preprocess_image,
    get_class_names,
    IMAGE_SIZE,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DATASET_PATH = 'dataset'
MODELS_PATH = 'models'
UPLOADS_PATH = 'uploads'
PREDICTIONS_CSV = os.path.join(UPLOADS_PATH, 'prediction_history.csv')

# Create uploads directory
os.makedirs(UPLOADS_PATH, exist_ok=True)


@st.cache_resource
def load_model():
    """Load the trained model with caching."""
    model_path = os.path.join(MODELS_PATH, 'best_model.keras')
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}")
        st.stop()
    return tf.keras.models.load_model(model_path)


@st.cache_data
def load_classes():
    """Load class names with caching."""
    return get_class_names(DATASET_PATH)


def predict_image(model, image_array: np.ndarray, class_names: list) -> dict:
    """
    Predict fruit class for an image array.
    
    Args:
        model: Trained model
        image_array (np.ndarray): Preprocessed image array
        class_names (list): List of class names
        
    Returns:
        dict: Prediction results
    """
    image_batch = np.expand_dims(image_array, axis=0)
    predictions = model.predict(image_batch, verbose=0)
    predicted_probs = predictions[0]
    
    predicted_idx = np.argmax(predicted_probs)
    predicted_class = class_names[predicted_idx]
    confidence = float(predicted_probs[predicted_idx])
    
    # Get top-3 predictions
    top_3_indices = np.argsort(predicted_probs)[::-1][:3]
    top_3_predictions = [
        {
            'class': class_names[idx],
            'confidence': float(predicted_probs[idx]),
            'percentage': float(predicted_probs[idx] * 100)
        }
        for idx in top_3_indices
    ]
    
    results = {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'confidence_percentage': confidence * 100,
        'top_3_predictions': top_3_predictions,
        'probabilities': {
            class_names[i]: float(predicted_probs[i])
            for i in range(len(class_names))
        }
    }
    
    return results


def save_prediction_history(image_name: str, predicted_class: str, 
                           confidence: float, method: str) -> None:
    """
    Save prediction history to CSV file.
    
    Args:
        image_name (str): Name of the image
        predicted_class (str): Predicted fruit class
        confidence (float): Confidence percentage
        method (str): Upload method ('file', 'webcam')
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create CSV file if it doesn't exist
    if not os.path.exists(PREDICTIONS_CSV):
        with open(PREDICTIONS_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Image Name', 'Predicted Class', 
                           'Confidence (%)', 'Method'])
    
    # Append prediction
    with open(PREDICTIONS_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, image_name, predicted_class, 
                        f'{confidence:.2f}', method])


def process_uploaded_file(uploaded_file) -> Optional[np.ndarray]:
    """
    Process uploaded image file.
    
    Args:
        uploaded_file: Streamlit uploaded file
        
    Returns:
        Optional[np.ndarray]: Preprocessed image array or None
    """
    try:
        image = Image.open(uploaded_file).convert('RGB')
        return load_and_preprocess_image(uploaded_file, IMAGE_SIZE)
    except Exception as e:
        logger.error(f"Error processing uploaded file: {str(e)}")
        return None


def process_webcam_frame(frame) -> Optional[np.ndarray]:
    """
    Process webcam frame.
    
    Args:
        frame: OpenCV frame
        
    Returns:
        Optional[np.ndarray]: Preprocessed image array or None
    """
    try:
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize
        resized = cv2.resize(rgb_frame, IMAGE_SIZE)
        # Normalize
        normalized = resized / 255.0
        return normalized
    except Exception as e:
        logger.error(f"Error processing webcam frame: {str(e)}")
        return None


def main():
    """Main Streamlit application."""
    # Page configuration
    st.set_page_config(
        page_title="🍎 Fruit Image Classifier",
        page_icon="🍎",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
            .main {
                padding: 2rem;
            }
            .prediction-box {
                padding: 1.5rem;
                border-radius: 0.5rem;
                background-color: #f0f2f6;
                margin: 1rem 0;
            }
            .confidence-high {
                color: #00c851;
                font-weight: bold;
            }
            .confidence-medium {
                color: #ffbb33;
                font-weight: bold;
            }
            .confidence-low {
                color: #ff4444;
                font-weight: bold;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Load model and classes
    model = load_model()
    class_names = load_classes()
    
    # Sidebar
    with st.sidebar:
        st.title("🔧 Settings")
        st.markdown("---")
        
        page = st.radio(
            "Select Mode:",
            ["📸 Image Upload", "🎥 Webcam Prediction", "📊 History"]
        )
        
        st.markdown("---")
        st.info(
            "🍎 This app classifies images of:\n"
            "- Apple\n"
            "- Banana\n"
            "- Orange"
        )
    
    # Main content
    st.title("🍎 Fruit Image Classifier")
    st.markdown("Using CNN Deep Learning Model")
    st.markdown("---")
    
    if page == "📸 Image Upload":
        image_upload_page(model, class_names)
    elif page == "🎥 Webcam Prediction":
        webcam_prediction_page(model, class_names)
    else:
        history_page()


def image_upload_page(model, class_names):
    """Image upload page."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image of Apple, Banana, or Orange"
        )
        
        if uploaded_file:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        st.subheader("🎯 Prediction Results")
        
        if uploaded_file:
            # Process image
            image_array = process_uploaded_file(uploaded_file)
            
            if image_array is not None:
                # Get prediction
                results = predict_image(model, image_array, class_names)
                
                # Display main prediction
                st.markdown(f"""
                    <div class="prediction-box">
                        <h3>Predicted Fruit: <span class="confidence-high">{results['predicted_class']}</span></h3>
                        <p style="font-size: 1.2em; margin: 0.5em 0;">
                            Confidence: <span class="confidence-high">{results['confidence_percentage']:.2f}%</span>
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Top-3 Predictions
                st.markdown("### Top 3 Predictions")
                for i, pred in enumerate(results['top_3_predictions'], 1):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.progress(pred['confidence'])
                    with col2:
                        st.markdown(f"<p style='text-align: center; margin-top: 0.5em;'><strong>{pred['class']}</strong><br>{pred['percentage']:.1f}%</p>", unsafe_allow_html=True)
                
                # All probabilities
                st.markdown("### All Class Probabilities")
                prob_data = pd.DataFrame([
                    {'Class': class_name, 'Probability': prob * 100}
                    for class_name, prob in results['probabilities'].items()
                ])
                st.bar_chart(prob_data.set_index('Class'))
                
                # Save to history
                save_prediction_history(
                    uploaded_file.name,
                    results['predicted_class'],
                    results['confidence_percentage'],
                    'file'
                )
                
                st.success("✅ Prediction saved to history!")
            else:
                st.error("❌ Error processing image. Please try another.")


def webcam_prediction_page(model, class_names):
    """Webcam prediction page."""
    st.subheader("🎥 Real-time Webcam Prediction")
    
    camera_image = st.camera_input("Capture an image with your webcam")
    
    if camera_image:
        # Convert to PIL Image
        image = Image.open(camera_image)
        
        # Process image
        image_array = process_uploaded_file(camera_image)
        
        if image_array is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Captured Image", use_column_width=True)
            
            with col2:
                # Get prediction
                results = predict_image(model, image_array, class_names)
                
                # Display results
                st.markdown(f"""
                    <div class="prediction-box">
                        <h3>Predicted Fruit: <span class="confidence-high">{results['predicted_class']}</span></h3>
                        <p style="font-size: 1.2em; margin: 0.5em 0;">
                            Confidence: <span class="confidence-high">{results['confidence_percentage']:.2f}%</span>
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Top-3 Predictions
                st.markdown("### Top 3 Predictions")
                for pred in results['top_3_predictions']:
                    st.progress(pred['confidence'])
                    st.text(f"{pred['class']}: {pred['percentage']:.2f}%")
                
                # Save to history
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_prediction_history(
                    f"webcam_{timestamp}.jpg",
                    results['predicted_class'],
                    results['confidence_percentage'],
                    'webcam'
                )
                
                st.success("✅ Prediction saved to history!")


def history_page():
    """Prediction history page."""
    st.subheader("📊 Prediction History")
    
    if os.path.exists(PREDICTIONS_CSV):
        df = pd.read_csv(PREDICTIONS_CSV)
        
        # Display statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Predictions", len(df))
        
        with col2:
            most_common = df['Predicted Class'].mode()[0] if len(df) > 0 else "N/A"
            st.metric("Most Predicted", most_common)
        
        with col3:
            avg_confidence = df['Confidence (%)'].mean() if len(df) > 0 else 0
            st.metric("Average Confidence", f"{avg_confidence:.2f}%")
        
        st.markdown("---")
        
        # Display history table
        st.markdown("### Prediction Records")
        st.dataframe(df, use_container_width=True)
        
        # Download history
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download History as CSV",
            data=csv,
            file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        # Clear history
        if st.button("🗑️ Clear History", help="This will delete all prediction history"):
            os.remove(PREDICTIONS_CSV)
            st.success("History cleared!")
            st.rerun()
    else:
        st.info("📭 No prediction history yet. Make some predictions to see them here!")


if __name__ == '__main__':
    main()