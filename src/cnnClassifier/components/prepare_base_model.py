from cnnClassifier.entity.config_entity import PrepareBaseModelConfig
import tensorflow as tf
from typing import List
from pathlib import Path

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weight,
            include_top=self.config.params_include_top,
        )

        self.save_model(
            self.config.base_model_path,
            self.model
        )

    @staticmethod
    def _prepare_full_model(model: tf.keras.Model, classes: int, freeze_all: bool, freeze_till: int,
                            learning_rate: float) -> tf.keras.Model:
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        elif freeze_till is not None:
            for layer in model.layers[:freeze_till]:
                layer.trainable = False
            for layer in model.layers[freeze_till:]:
                layer.trainable = True

        x = model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        predictions = tf.keras.layers.Dense(classes, activation='softmax')(x)

        full_model = tf.keras.Model(inputs=model.input, outputs=predictions)

        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy']
        )

        return full_model

    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(
            self.config.updated_model_path,
            self.full_model
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
