"""Fast and accurate fruit classifier training using MobileNetV2 transfer learning."""

import json
import os
import logging
from typing import Tuple

from tensorflow import keras
from tensorflow.keras import layers

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
EPOCHS = 10


def build_mobilenet_model(num_classes: int) -> keras.Model:
    """Build a compact MobileNetV2 transfer-learning classifier.

    The ImageNet backbone is frozen for fast CPU training. The Rescaling
    layer converts the existing [0, 1] dataset values to the [-1, 1]
    range expected by MobileNetV2, so the Streamlit preprocessing remains
    compatible without requiring a separate deployment change.
    """
    inputs = keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), name='image')

    # Dataset generator already rescales pixels to [0, 1].
    x = layers.Rescaling(2.0, offset=-1.0, name='mobilenet_input_scaling')(inputs)

    try:
        base_model = keras.applications.MobileNetV2(
            input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
            include_top=False,
            weights='imagenet',
        )
    except Exception as exc:
        logger.warning(
            "Could not download ImageNet weights (%s). Falling back to random weights.",
            exc,
        )
        base_model = keras.applications.MobileNetV2(
            input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
            include_top=False,
            weights=None,
        )

    base_model.trainable = False
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name='global_average_pooling')(x)
    x = layers.Dense(128, activation='relu', name='classifier_dense')(x)
    x = layers.Dropout(0.30, name='classifier_dropout')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    return keras.Model(inputs, outputs, name='FruitClassifierMobileNetV2')


def compile_model(model: keras.Model) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def get_callbacks(model_save_path: str) -> list:
    return [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


def train_model(
    train_dataset,
    validation_dataset,
    epochs: int = EPOCHS,
) -> Tuple[keras.Model, dict]:
    num_classes = len(train_dataset.class_indices)
    model = compile_model(build_mobilenet_model(num_classes))
    model.summary()

    model_save_path = os.path.join(MODELS_PATH, 'best_model.keras')
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=get_callbacks(model_save_path),
        verbose=1,
    )
    log_message('MobileNetV2 transfer-learning training completed successfully!')
    return model, history.history


def save_class_names(train_dataset) -> None:
    class_names = [None] * len(train_dataset.class_indices)
    for name, index in train_dataset.class_indices.items():
        class_names[index] = name

    path = os.path.join(MODELS_PATH, 'class_names.json')
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(class_names, file, indent=2)
    logger.info('Class mapping saved to %s: %s', path, class_names)


def save_model(model: keras.Model, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    logger.info('Model saved to %s', save_path)


def main():
    create_directories([MODELS_PATH, RESULTS_PATH])
    logger.info('Loading datasets...')
    train_dataset = load_dataset(
        DATASET_PATH,
        subset='train',
        batch_size=BATCH_SIZE,
    )
    validation_dataset = load_dataset(
        DATASET_PATH,
        subset='validation',
        batch_size=BATCH_SIZE,
    )

    model, history = train_model(
        train_dataset,
        validation_dataset,
        epochs=EPOCHS,
    )
    save_class_names(train_dataset)
    save_model(model, os.path.join(MODELS_PATH, 'fruit_classifier.keras'))
    plot_training_history(
        history,
        os.path.join(RESULTS_PATH, 'training_history.png'),
    )
    logger.info('Training pipeline completed successfully!')


if __name__ == '__main__':
    main()
