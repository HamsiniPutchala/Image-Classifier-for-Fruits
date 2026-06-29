"""Evaluation script for trained fruit classifier model."""

import os
import logging
import numpy as np
import tensorflow as tf
from typing import Tuple

from utils import (
    load_dataset,
    get_class_names,
    calculate_metrics,
    print_metrics,
    get_predictions_from_probabilities,
    plot_confusion_matrix,
    print_classification_report,
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
RESULTS_PATH = 'results'
BATCH_SIZE = 32


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


def get_predictions(
    model: tf.keras.Model,
    test_dataset
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get predictions on test dataset.
    
    Args:
        model (tf.keras.Model): Trained model
        test_dataset: Test dataset
        
    Returns:
        Tuple: (true_labels, predicted_labels, probabilities)
    """
    logger.info("Getting predictions on test dataset...")
    
    true_labels = []
    predicted_labels = []
    probabilities = []
    
    for images, labels in test_dataset:
        probs = model.predict(images, verbose=0)
        probabilities.extend(probs)
        
        preds, _ = get_predictions_from_probabilities(probs)
        predicted_labels.extend(preds)
        
        true_labels.extend(np.argmax(labels.numpy(), axis=1))
    
    return (
        np.array(true_labels),
        np.array(predicted_labels),
        np.array(probabilities)
    )


def evaluate_model(
    model: tf.keras.Model,
    test_dataset,
    class_names: list
) -> dict:
    """
    Comprehensive model evaluation.
    
    Args:
        model (tf.keras.Model): Trained model
        test_dataset: Test dataset
        class_names (list): List of class names
        
    Returns:
        dict: Evaluation results
    """
    # Get predictions
    true_labels, predicted_labels, probabilities = get_predictions(model, test_dataset)
    
    # Calculate metrics
    metrics = calculate_metrics(true_labels, predicted_labels)
    
    # Print metrics
    print_metrics(metrics)
    
    # Print classification report
    print_classification_report(true_labels, predicted_labels, class_names)
    
    # Plot confusion matrix
    confusion_matrix_path = os.path.join(RESULTS_PATH, 'confusion_matrix.png')
    plot_confusion_matrix(true_labels, predicted_labels, class_names, confusion_matrix_path)
    
    return {
        'true_labels': true_labels,
        'predicted_labels': predicted_labels,
        'probabilities': probabilities,
        'metrics': metrics,
        'class_names': class_names
    }


def print_detailed_evaluation(results: dict) -> None:
    """
    Print detailed evaluation summary.
    
    Args:
        results (dict): Evaluation results dictionary
    """
    print("\n" + "="*70)
    print("DETAILED MODEL EVALUATION REPORT")
    print("="*70)
    
    metrics = results['metrics']
    print(f"\nTest Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision:      {metrics['precision']:.4f}")
    print(f"Recall:         {metrics['recall']:.4f}")
    print(f"F1 Score:       {metrics['f1_score']:.4f}")
    
    print("\n" + "="*70 + "\n")


def main():
    """Main evaluation pipeline."""
    # Load class names
    class_names = get_class_names(DATASET_PATH)
    
    # Load test dataset
    logger.info("Loading test dataset...")
    test_dataset = load_dataset(DATASET_PATH, subset='test', batch_size=BATCH_SIZE)
    
    # Load model
    model_path = os.path.join(MODELS_PATH, 'best_model.keras')
    model = load_trained_model(model_path)
    
    # Evaluate model
    results = evaluate_model(model, test_dataset, class_names)
    
    # Print detailed evaluation
    print_detailed_evaluation(results)
    
    logger.info("Evaluation completed successfully!")


if __name__ == '__main__':
    main()