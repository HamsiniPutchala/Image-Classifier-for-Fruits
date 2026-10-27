"""Evaluate the trained fruit classifier on the held-out validation split."""

import json
import logging
import os
from typing import Tuple

import numpy as np
import tensorflow as tf

from utils import (
    load_dataset,
    calculate_metrics,
    print_metrics,
    get_predictions_from_probabilities,
    plot_confusion_matrix,
    print_classification_report,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DATASET_PATH = 'dataset'
MODELS_PATH = 'models'
RESULTS_PATH = 'results'
BATCH_SIZE = 32


def load_class_names() -> list[str]:
    metadata_path = os.path.join(MODELS_PATH, 'class_names.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as file:
            names = json.load(file)
        if isinstance(names, list) and names:
            return names

    train_path = os.path.join(DATASET_PATH, 'train')
    if not os.path.isdir(train_path):
        raise FileNotFoundError(f"Training dataset not found: {train_path}")
    return sorted(
        name for name in os.listdir(train_path)
        if os.path.isdir(os.path.join(train_path, name))
    )


def load_trained_model(model_path: str) -> tf.keras.Model:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    return tf.keras.models.load_model(model_path)


def get_predictions(
    model: tf.keras.Model,
    validation_dataset,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    true_labels = []
    predicted_labels = []
    probabilities = []

    for images, labels in validation_dataset:
        probs = model.predict(images, verbose=0)
        probabilities.extend(probs)
        preds, _ = get_predictions_from_probabilities(probs)
        predicted_labels.extend(preds)
        true_labels.extend(np.argmax(labels.numpy(), axis=1))

    return np.array(true_labels), np.array(predicted_labels), np.array(probabilities)


def main() -> None:
    os.makedirs(RESULTS_PATH, exist_ok=True)
    class_names = load_class_names()
    validation_dataset = load_dataset(
        DATASET_PATH, subset='validation', batch_size=BATCH_SIZE
    )
    validation_dataset.reset()
    model = load_trained_model(os.path.join(MODELS_PATH, 'best_model.keras'))

    true_labels, predicted_labels, _ = get_predictions(model, validation_dataset)
    metrics = calculate_metrics(true_labels, predicted_labels)
    print_metrics(metrics)
    print_classification_report(true_labels, predicted_labels, class_names)
    plot_confusion_matrix(
        true_labels,
        predicted_labels,
        class_names,
        os.path.join(RESULTS_PATH, 'confusion_matrix.png'),
    )

    logger.info(
        "Validation accuracy: %.2f%% | F1: %.4f",
        metrics['accuracy'] * 100,
        metrics['f1_score'],
    )


if __name__ == '__main__':
    main()
