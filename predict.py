"""Prediction script for single image classification."""

import os
import sys
import logging
import numpy as np
import tensorflow as tf
from typing import Tuple, List, Dict

from utils import (
    load_and_preprocess_image,
    get_class_names,
    IMAGE_SIZE,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DATASET_PATH = 'dataset'
MODELS_PATH = 'models'


def load_trained_model(model_path: str) -> tf.keras.Model:
    """
    Load a trained model.
    
    Args:
        model_path (str): Path to the trained model
        
    Returns:
        tf.keras.Model: Loaded model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = tf.keras.models.load_model(model_path)
    logger.info(f"Model loaded from {model_path}")
    return model


def predict_single_image(
    model: tf.keras.Model,
    image_path: str,
    class_names: List[str]
) -> Dict:
    """
    Predict fruit class for a single image.
    
    Args:
        model (tf.keras.Model): Trained model
        image_path (str): Path to input image
        class_names (List[str]): List of class names
        
    Returns:
        Dict: Prediction results including class, confidence, and top-3 predictions
    """
    # Load and preprocess image
    image_array = load_and_preprocess_image(image_path, IMAGE_SIZE)
    
    if image_array is None:
        raise ValueError(f"Failed to load image from {image_path}")
    
    # Expand dimensions for batch
    image_batch = np.expand_dims(image_array, axis=0)
    
    # Get predictions
    predictions = model.predict(image_batch, verbose=0)
    predicted_probs = predictions[0]
    
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
    
    # Get predicted class
    predicted_idx = np.argmax(predicted_probs)
    predicted_class = class_names[predicted_idx]
    confidence = float(predicted_probs[predicted_idx])
    
    results = {
        'image_path': image_path,
        'predicted_class': predicted_class,
        'confidence': confidence,
        'confidence_percentage': confidence * 100,
        'top_3_predictions': top_3_predictions,
        'all_probabilities': {
            class_names[i]: float(predicted_probs[i])
            for i in range(len(class_names))
        }
    }
    
    return results


def print_prediction_results(results: Dict) -> None:
    """
    Print prediction results in formatted manner.
    
    Args:
        results (Dict): Prediction results from predict_single_image()
    """
    print("\n" + "="*70)
    print("PREDICTION RESULTS")
    print("="*70)
    
    print(f"\nImage: {results['image_path']}")
    print(f"\nPredicted Class: {results['predicted_class']}")
    print(f"Confidence: {results['confidence_percentage']:.2f}%")
    
    print("\nTop-3 Predictions:")
    print("-" * 70)
    for i, pred in enumerate(results['top_3_predictions'], 1):
        print(f"{i}. {pred['class']:<15} - {pred['percentage']:>6.2f}%")
    
    print("\nAll Class Probabilities:")
    print("-" * 70)
    for class_name, prob in results['all_probabilities'].items():
        bar_length = int(prob * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        print(f"{class_name:<15} [{bar}] {prob*100:6.2f}%")
    
    print("\n" + "="*70 + "\n")


def predict_batch(
    model: tf.keras.Model,
    image_paths: List[str],
    class_names: List[str]
) -> List[Dict]:
    """
    Predict for multiple images.
    
    Args:
        model (tf.keras.Model): Trained model
        image_paths (List[str]): List of image paths
        class_names (List[str]): List of class names
        
    Returns:
        List[Dict]: List of prediction results
    """
    results = []
    for image_path in image_paths:
        try:
            result = predict_single_image(model, image_path, class_names)
            results.append(result)
            print_prediction_results(result)
        except Exception as e:
            logger.error(f"Error predicting for {image_path}: {str(e)}")
    
    return results


def main():
    """Main prediction pipeline."""
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        print("Example: python predict.py dataset/test/Apple/apple_001.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Load class names
    class_names = get_class_names(DATASET_PATH)
    
    # Load model
    model_path = os.path.join(MODELS_PATH, 'best_model.keras')
    model = load_trained_model(model_path)
    
    # Predict
    results = predict_single_image(model, image_path, class_names)
    
    # Print results
    print_prediction_results(results)
    
    logger.info("Prediction completed successfully!")


if __name__ == '__main__':
    main()