"""Evaluate the trained fruit classifier on the validation split."""

import json
import logging
import os

import numpy as np
import tensorflow as tf

from utils import (
    load_dataset,
    calculate_metrics,
    print_metrics,
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


def load_class_names():
    metadata_path = os.path.join(MODELS_PATH, 'class_names.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as file:
            names = json.load(file)
        if isinstance(names, list) and names:
            return names

    train_path = os.path.join(DATASET_PATH, 'train')
    if not os.path.isdir(train_path):
        raise FileNotFoundError(f'Training dataset not found: {train_path}')
    return sorted(
        name for name in os.listdir(train_path)
        if os.path.isdir(os.path.join(train_path, name))
    )


def load_trained_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model not found at {model_path}')
    return tf.keras.models.load_model(model_path, compile=False)


def get_predictions(model, validation_dataset):
    """Evaluate exactly one complete validation pass."""
    validation_dataset.reset()

    probabilities = np.asarray(model.predict(validation_dataset, verbose=0))
    true_labels = np.asarray(validation_dataset.classes, dtype=np.int64)

    if probabilities.ndim != 2:
        raise RuntimeError(
            f'Unexpected model output shape: {probabilities.shape}. Expected 2-D probabilities.'
        )

    if len(true_labels) != len(probabilities):
        raise RuntimeError(
            f'Validation size mismatch: {len(true_labels)} labels vs '
            f'{len(probabilities)} predictions.'
        )

    if probabilities.shape[1] != len(validation_dataset.class_indices):
        raise RuntimeError(
            f'Model output count ({probabilities.shape[1]}) does not match '
            f'dataset class count ({len(validation_dataset.class_indices)}).'
        )

    predicted_labels = np.argmax(probabilities, axis=1).astype(np.int64)
    return true_labels, predicted_labels, probabilities


def main():
    os.makedirs(RESULTS_PATH, exist_ok=True)

    class_names = load_class_names()
    if len(class_names) < 2:
        raise RuntimeError(f'Expected at least 2 fruit classes, found {len(class_names)}: {class_names}')

    validation_dataset = load_dataset(
        DATASET_PATH,
        subset='validation',
        batch_size=BATCH_SIZE,
    )
    model = load_trained_model(os.path.join(MODELS_PATH, 'best_model.keras'))

    if len(class_names) != len(validation_dataset.class_indices):
        raise RuntimeError(
            f'Class metadata count ({len(class_names)}) does not match '
            f'validation dataset count ({len(validation_dataset.class_indices)}).'
        )

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
        'Validation accuracy: %.2f%% | F1: %.4f | Classes: %d',
        metrics['accuracy'] * 100,
        metrics['f1_score'],
        len(class_names),
    )


if __name__ == '__main__':
    main()
