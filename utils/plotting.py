"""Plotting and visualization utilities."""

import logging
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from typing import Dict, List

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 5)


def plot_training_history(
    history: Dict,
    save_path: str = 'results/training_history.png'
) -> None:
    """
    Plot training and validation accuracy and loss.
    
    Args:
        history (Dict): Training history from model.fit()
        save_path (str): Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    axes[0].plot(history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[0].plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    
    # Loss
    axes[1].plot(history['loss'], label='Training Loss', linewidth=2)
    axes[1].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Training history plot saved to {save_path}")
    plt.close()


def plot_accuracy(
    history: Dict,
    save_path: str = 'results/accuracy.png'
) -> None:
    """
    Plot training and validation accuracy.
    
    Args:
        history (Dict): Training history from model.fit()
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    plt.plot(history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    plt.title('Model Accuracy Over Epochs', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Accuracy plot saved to {save_path}")
    plt.close()


def plot_loss(
    history: Dict,
    save_path: str = 'results/loss.png'
) -> None:
    """
    Plot training and validation loss.
    
    Args:
        history (Dict): Training history from model.fit()
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], 'b-', label='Training Loss', linewidth=2)
    plt.plot(history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    plt.title('Model Loss Over Epochs', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Loss plot saved to {save_path}")
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: str = 'results/confusion_matrix.png'
) -> None:
    """
    Plot and save confusion matrix.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        class_names (List[str]): List of class names
        save_path (str): Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Confusion matrix saved to {save_path}")
    plt.close()


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str]
) -> str:
    """
    Generate and print classification report.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        class_names (List[str]): List of class names
        
    Returns:
        str: Classification report as string
    """
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(report)
    return report