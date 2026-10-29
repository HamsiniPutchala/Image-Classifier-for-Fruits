"""Accurate 10-class fruit classifier training with MobileNetV2 fine-tuning."""

import json
import os
import logging
from typing import Tuple

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from utils import load_dataset, create_directories, plot_training_history, log_message

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATASET_PATH = 'dataset'
MODELS_PATH = 'models'
RESULTS_PATH = 'results'
IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32
INITIAL_EPOCHS = 5
FINE_TUNE_EPOCHS = 15
FINE_TUNE_LAYERS = 60


def build_mobilenet_model(num_classes: int) -> keras.Model:
    inputs = keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), name='image')
    x = layers.Rescaling(2.0, offset=-1.0, name='mobilenet_input_scaling')(inputs)
    try:
        base_model = keras.applications.MobileNetV2(
            input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), include_top=False, weights='imagenet'
        )
    except Exception as exc:
        logger.warning('Could not download ImageNet weights (%s). Falling back to random weights.', exc)
        base_model = keras.applications.MobileNetV2(
            input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), include_top=False, weights=None
        )
    base_model._name = 'mobilenetv2_backbone'
    base_model.trainable = False
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name='global_average_pooling')(x)
    x = layers.Dense(128, activation='relu', name='classifier_dense')(x)
    x = layers.Dropout(0.30, name='classifier_dropout')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    return keras.Model(inputs, outputs, name='FruitClassifierMobileNetV2')


def compile_model(model: keras.Model, learning_rate: float) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=['accuracy'],
    )
    return model


def callbacks(model_save_path: str) -> list:
    return [
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=4, mode='max', restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(model_save_path, monitor='val_accuracy', mode='max', save_best_only=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7, verbose=1),
    ]


def train_model(train_dataset, validation_dataset, epochs: int = INITIAL_EPOCHS) -> Tuple[keras.Model, dict]:
    num_classes = len(train_dataset.class_indices)
    model = build_mobilenet_model(num_classes)
    model_save_path = os.path.join(MODELS_PATH, 'best_model.keras')
    compile_model(model, 1e-3)
    logger.info('Stage 1: training classification head with frozen MobileNetV2 backbone')
    first = model.fit(train_dataset, validation_data=validation_dataset, epochs=epochs, callbacks=callbacks(model_save_path), verbose=1)

    backbone = model.get_layer('mobilenetv2_backbone')
    backbone.trainable = True
    for layer in backbone.layers[:-FINE_TUNE_LAYERS]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    compile_model(model, 1e-5)
    logger.info('Stage 2: fine-tuning last %d MobileNetV2 layers', FINE_TUNE_LAYERS)
    second = model.fit(train_dataset, validation_data=validation_dataset, epochs=FINE_TUNE_EPOCHS, callbacks=callbacks(model_save_path), verbose=1)
    history = {key: first.history.get(key, []) + second.history.get(key, []) for key in set(first.history) | set(second.history)}
    log_message('MobileNetV2 fine-tuning training completed successfully!')
    return model, history


def save_class_names(train_dataset) -> list[str]:
    class_names = [None] * len(train_dataset.class_indices)
    for name, index in train_dataset.class_indices.items():
        class_names[index] = name
    with open(os.path.join(MODELS_PATH, 'class_names.json'), 'w', encoding='utf-8') as file:
        json.dump(class_names, file, indent=2)
    return class_names


def save_ood_calibration(model: keras.Model, validation_dataset, class_names: list[str]) -> None:
    embedding_model = keras.Model(model.input, model.get_layer('global_average_pooling').output)
    validation_dataset.reset()
    embeddings = np.asarray(embedding_model.predict(validation_dataset, verbose=0), dtype=np.float32)
    labels = np.asarray(validation_dataset.classes, dtype=np.int64)
    centroids, thresholds = [], []
    for index in range(len(class_names)):
        class_embeddings = embeddings[labels == index]
        if len(class_embeddings) == 0:
            raise RuntimeError(f'No validation embeddings found for class index {index}')
        centroid = class_embeddings.mean(axis=0)
        class_distances = np.linalg.norm(class_embeddings - centroid, axis=1)
        centroids.append(centroid)
        thresholds.append(float(np.percentile(class_distances, 99.0) * 1.25))
    np.savez_compressed(
        os.path.join(MODELS_PATH, 'ood_calibration.npz'),
        centroids=np.asarray(centroids, dtype=np.float32),
        distance_thresholds=np.asarray(thresholds, dtype=np.float32),
        distance_threshold=np.asarray(float(np.max(thresholds)), dtype=np.float32),
    )
    logger.info('Saved class-specific OOD thresholds: %s', [round(x, 4) for x in thresholds])


def save_model(model: keras.Model, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)


def main():
    create_directories([MODELS_PATH, RESULTS_PATH])
    train_dataset = load_dataset(DATASET_PATH, subset='train', batch_size=BATCH_SIZE)
    validation_dataset = load_dataset(DATASET_PATH, subset='validation', batch_size=BATCH_SIZE)
    _, history = train_model(train_dataset, validation_dataset)
    best_model_path = os.path.join(MODELS_PATH, 'best_model.keras')
    best_model = keras.models.load_model(best_model_path, compile=False)
    class_names = save_class_names(train_dataset)
    save_ood_calibration(best_model, validation_dataset, class_names)
    save_model(best_model, os.path.join(MODELS_PATH, 'fruit_classifier.keras'))
    plot_training_history(history, os.path.join(RESULTS_PATH, 'training_history.png'))
    logger.info('Training pipeline completed successfully with the best validation checkpoint.')


if __name__ == '__main__':
    main()
