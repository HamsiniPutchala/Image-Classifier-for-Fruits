"""Training script for fruit image classifier CNN model."""

import json
import os
import logging
from typing import Tuple

from tensorflow import keras
from tensorflow.keras import layers, models

from utils import (
    load_dataset,
    create_directories,
    plot_training_history,
    log_message,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DATASET_PATH = 'dataset'
MODELS_PATH = 'models'
RESULTS_PATH = 'results'
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 20


def build_cnn_model(num_classes: int) -> keras.Model:
    """Build a compact CNN with strong feature extraction and a small model size."""
    model = models.Sequential([
        layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),

        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.20),

        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),

        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.30),

        layers.Conv2D(192, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.30),

        # GlobalAveragePooling avoids the very large Flatten -> Dense layer.
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.40),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def compile_model(model: keras.Model) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def get_callbacks(model_save_path: str) -> list:
    return [
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            model_save_path, monitor='val_accuracy', save_best_only=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
    ]


def train_model(train_dataset, validation_dataset, epochs: int = EPOCHS) -> Tuple[keras.Model, dict]:
    num_classes = len(train_dataset.class_indices)
    model = compile_model(build_cnn_model(num_classes))
    model.summary()

    model_save_path = os.path.join(MODELS_PATH, 'best_model.keras')
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=get_callbacks(model_save_path),
        verbose=1,
    )
    log_message("Training completed successfully!")
    return model, history.history


def save_class_names(train_dataset) -> None:
    class_names = [None] * len(train_dataset.class_indices)
    for name, index in train_dataset.class_indices.items():
        class_names[index] = name
    path = os.path.join(MODELS_PATH, 'class_names.json')
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(class_names, file, indent=2)
    logger.info("Class mapping saved to %s: %s", path, class_names)


def save_model(model: keras.Model, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    logger.info("Model saved to %s", save_path)


def main():
    create_directories([MODELS_PATH, RESULTS_PATH])
    logger.info("Loading datasets...")
    train_dataset = load_dataset(DATASET_PATH, subset='train', batch_size=BATCH_SIZE)
    validation_dataset = load_dataset(DATASET_PATH, subset='validation', batch_size=BATCH_SIZE)

    model, history = train_model(train_dataset, validation_dataset, epochs=EPOCHS)
    save_class_names(train_dataset)
    save_model(model, os.path.join(MODELS_PATH, 'fruit_classifier.keras'))
    plot_training_history(history, os.path.join(RESULTS_PATH, 'training_history.png'))
    logger.info("Training pipeline completed successfully!")


if __name__ == '__main__':
    main()
