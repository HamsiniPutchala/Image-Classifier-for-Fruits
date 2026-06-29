"""Image preprocessing utilities for fruit classification."""

import os
import logging
from typing import Tuple, Optional
import numpy as np
import tensorflow as tf
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2


def load_and_preprocess_image(
    image_path: str,
    image_size: Tuple[int, int] = IMAGE_SIZE
) -> Optional[np.ndarray]:
    """
    Load and preprocess a single image.
    
    Args:
        image_path (str): Path to the image file
        image_size (Tuple[int, int]): Target image size
        
    Returns:
        Optional[np.ndarray]: Normalized image array or None if loading fails
    """
    try:
        image = Image.open(image_path).convert('RGB')
        image = image.resize(image_size, Image.Resampling.LANCZOS)
        image_array = np.array(image) / 255.0
        return image_array
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {str(e)}")
        return None


def create_data_generators(
    image_size: Tuple[int, int] = IMAGE_SIZE
) -> Tuple[tf.keras.preprocessing.image.ImageDataGenerator,
           tf.keras.preprocessing.image.ImageDataGenerator]:
    """
    Create training and validation data generators with augmentation.
    
    Args:
        image_size (Tuple[int, int]): Target image size
        
    Returns:
        Tuple: (train_generator, validation_generator)
    """
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        fill_mode='nearest'
    )
    
    validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255.0
    )
    
    return train_datagen, validation_datagen


def load_dataset(
    dataset_path: str,
    subset: str = 'train',
    image_size: Tuple[int, int] = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE
) -> tf.data.Dataset:
    """
    Load dataset from directory structure.
    
    Args:
        dataset_path (str): Path to dataset directory
        subset (str): 'train', 'validation', or 'test'
        image_size (Tuple[int, int]): Target image size
        batch_size (int): Batch size for data loading
        
    Returns:
        tf.data.Dataset: TensorFlow dataset
    """
    subset_path = os.path.join(dataset_path, subset)
    
    if not os.path.exists(subset_path):
        logger.error(f"Dataset path not found: {subset_path}")
        raise ValueError(f"Dataset path not found: {subset_path}")
    
    if subset == 'train':
        train_datagen, _ = create_data_generators(image_size)
        dataset = train_datagen.flow_from_directory(
            subset_path,
            target_size=image_size,
            batch_size=batch_size,
            class_mode='categorical'
        )
    else:
        _, val_datagen = create_data_generators(image_size)
        dataset = val_datagen.flow_from_directory(
            subset_path,
            target_size=image_size,
            batch_size=batch_size,
            class_mode='categorical'
        )
    
    logger.info(f"Loaded {subset} dataset from {subset_path}")
    return dataset


def get_class_names(dataset_path: str) -> list:
    """
    Get class names from dataset directory structure.
    
    Args:
        dataset_path (str): Path to dataset directory
        
    Returns:
        list: List of class names sorted alphabetically
    """
    train_path = os.path.join(dataset_path, 'train')
    class_names = sorted([d for d in os.listdir(train_path)
                         if os.path.isdir(os.path.join(train_path, d))])
    logger.info(f"Found classes: {class_names}")
    return class_names