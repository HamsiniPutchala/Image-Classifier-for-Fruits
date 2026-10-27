"""Plotting and visualization utilities."""

import logging
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 5)


def plot_training_history(history: Dict, save_path: str = "results/training_history.png") -> None:
    """Save training/validation accuracy and loss plots."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.get("accuracy", []), label="Training Accuracy", linewidth=2)
    axes[0].plot(history.get("val_accuracy", []), label="Validation Accuracy", linewidth=2)
    axes[0].set_title("Model Accuracy", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.get("loss", []), label="Training Loss", linewidth=2)
    axes[1].plot(history.get("val_loss", []), label="Validation Loss", linewidth=2)
    axes[1].set_title("Model Loss", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    logger.info("Training history plot saved to %s", save_path)
    plt.close(fig)


def plot_accuracy(history: Dict, save_path: str = "results/accuracy.png") -> None:
    """Save training and validation accuracy plot."""
    plt.figure(figsize=(10, 6))
    plt.plot(history.get("accuracy", []), label="Training Accuracy", linewidth=2)
    plt.plot(history.get("val_accuracy", []), label="Validation Accuracy", linewidth=2)
    plt.title("Model Accuracy Over Epochs", fontsize=16, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_loss(history: Dict, save_path: str = "results/loss.png") -> None:
    """Save training and validation loss plot."""
    plt.figure(figsize=(10, 6))
    plt.plot(history.get("loss", []), label="Training Loss", linewidth=2)
    plt.plot(history.get("val_loss", []), label="Validation Loss", linewidth=2)
    plt.title("Model Loss Over Epochs", fontsize=16, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: str = "results/confusion_matrix.png",
) -> None:
    """Save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Count"},
    )
    plt.title("Confusion Matrix", fontsize=16, fontweight="bold")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> str:
    """Generate and print a classification report."""
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(report)
    return report
