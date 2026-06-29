"""Metrics calculation utilities."""

import logging
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = 'weighted'
) -> Dict[str, float]:
    """
    Calculate various evaluation metrics.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        average (str): Averaging method for multi-class metrics
        
    Returns:
        Dict[str, float]: Dictionary containing various metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    return metrics


def print_metrics(metrics: Dict[str, float]) -> None:
    """
    Print metrics in a formatted manner.
    
    Args:
        metrics (Dict[str, float]): Dictionary of metrics
    """
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name.replace('_', ' ').title()}: {metric_value:.4f}")
    print("="*60 + "\n")


def get_predictions_from_probabilities(
    probabilities: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert probability predictions to class predictions.
    
    Args:
        probabilities (np.ndarray): Probability matrix from model
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (predicted_classes, max_probabilities)
    """
    predicted_classes = np.argmax(probabilities, axis=1)
    max_probabilities = np.max(probabilities, axis=1)
    return predicted_classes, max_probabilities