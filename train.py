"""Training script for fruit image classifier CNN model."""

import os
import logging
from typing import Tuple
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

from utils import (
    load_dataset,
    get_class_names,
    create_directories,
    plot_training_history,
    log_message,
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
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 50


def build_cnn_model(num_classes: int) -> keras.Model:
    """
    Build CNN model for fruit classification.
    
    Args:
        num_classes (int): Number of output classes
        
    Returns:
        keras.Model: Compiled CNN model
    """
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), padding='same', activation='relu',
                     input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 2
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 3
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 4
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fully Connected Layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def compile_model(model: keras.Model) -> keras.Model:
    """
    Compile the model with optimizer, loss, and metrics.
    
    Args:
        model (keras.Model): Uncompiled model
        
    Returns:
        keras.Model: Compiled model
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    logger.info("Model compiled successfully")
    return model


def get_callbacks(model_save_path: str) -> list:
    """
    Create training callbacks.
    
    Args:
        model_save_path (str): Path to save the best model
        
    Returns:
        list: List of callbacks
    """
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
    ]
    return callbacks


def train_model(
    train_dataset,
    validation_dataset,
    epochs: int = EPOCHS
) -> Tuple[keras.Model, dict]:
    """
    Train the CNN model.
    
    Args:
        train_dataset: Training dataset
        validation_dataset: Validation dataset
        epochs (int): Number of training epochs
        
    Returns:
        Tuple[keras.Model, dict]: Trained model and history
    """
    num_classes = len(train_dataset.class_indices)
    model = build_cnn_model(num_classes)
    model = compile_model(model)
    
    logger.info(f"Model architecture:")
    model.summary()
    
    model_save_path = os.path.join(MODELS_PATH, 'best_model.keras')
    callbacks = get_callbacks(model_save_path)
    
    log_message("Starting model training...")
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    log_message("Training completed successfully!")
    return model, history.history


def save_model(model: keras.Model, save_path: str) -> None:
    """
    Save the trained model.
    
    Args:
        model (keras.Model): Trained model
        save_path (str): Path to save the model
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    logger.info(f"Model saved to {save_path}")


def main():
    """Main training pipeline."""
    # Create directories
    create_directories([MODELS_PATH, RESULTS_PATH])
    
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = load_dataset(DATASET_PATH, subset='train', batch_size=BATCH_SIZE)
    validation_dataset = load_dataset(DATASET_PATH, subset='validation', batch_size=BATCH_SIZE)
    
    # Train model
    model, history = train_model(train_dataset, validation_dataset, epochs=EPOCHS)
    
    # Save model
    final_model_path = os.path.join(MODELS_PATH, 'fruit_classifier.keras')
    save_model(model, final_model_path)
    
    # Plot training history
    plot_training_history(history, os.path.join(RESULTS_PATH, 'training_history.png'))
    
    logger.info("Training pipeline completed successfully!")


if __name__ == '__main__':
    main()