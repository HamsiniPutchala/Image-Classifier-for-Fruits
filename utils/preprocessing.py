"""Image preprocessing utilities for fruit classification."""

import logging
import os
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2


def load_and_preprocess_image(
    image_path: str,
    image_size: Tuple[int, int] = IMAGE_SIZE,
) -> Optional[np.ndarray]:
    """Load a single image and return an RGB float array in [0, 1]."""
    try:
        image = Image.open(image_path).convert('RGB')
        image = image.resize(image_size, Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.float32) / 255.0
    except Exception as exc:
        logger.error('Error loading image %s: %s', image_path, exc)
        return None


def create_data_generators(
    image_size: Tuple[int, int] = IMAGE_SIZE,
) -> Tuple[
    tf.keras.preprocessing.image.ImageDataGenerator,
    tf.keras.preprocessing.image.ImageDataGenerator,
]:
    """Create training and validation image generators."""
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        fill_mode='nearest',
    )
    validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255.0,
    )
    return train_datagen, validation_datagen


def load_dataset(
    dataset_path: str,
    subset: str = 'train',
    image_size: Tuple[int, int] = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
):
    """Load a train or validation DirectoryIterator."""
    subset_path = os.path.join(dataset_path, subset)
    if not os.path.isdir(subset_path):
        raise ValueError(f'Dataset path not found: {subset_path}')

    train_datagen, validation_datagen = create_data_generators(image_size)

    if subset == 'train':
        return train_datagen.flow_from_directory(
            subset_path,
            target_size=image_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True,
            seed=42,
        )

    if subset in ('validation', 'test'):
        return validation_datagen.flow_from_directory(
            subset_path,
            target_size=image_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False,
        )

    raise ValueError(f"Unsupported subset '{subset}'. Use 'train', 'validation', or 'test'.")


def get_class_names(dataset_path: str) -> list:
    """Return class names from the training directory."""
    train_path = os.path.join(dataset_path, 'train')
    if not os.path.isdir(train_path):
        raise ValueError(f'Training dataset path not found: {train_path}')
    class_names = sorted(
        directory
        for directory in os.listdir(train_path)
        if os.path.isdir(os.path.join(train_path, directory))
    )
    logger.info('Found classes: %s', class_names)
    return class_names
