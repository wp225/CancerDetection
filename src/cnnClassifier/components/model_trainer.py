import tensorflow as tf
from cnnClassifier.config.configuration import TrainingConfig
from pathlib import Path
from typing import List
from cnnClassifier import logger
class Train:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None

    def get_base_model(self):
        try:
            self.model = tf.keras.models.load_model(self.config.updated_base_model_path)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")

    def train_valid_generator(self):
        seed = 132
        image_size = tuple(self.config.params_image_size[:2])

        try:
            logger.info('Trying train generator')

            data_augmentation = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal_and_vertical"),
                tf.keras.layers.RandomRotation(0.2),
                tf.keras.layers.RandomZoom(0.2),
            ]) if self.config.params_is_augmented else None

            def preprocess(image, label):
                if data_augmentation:
                    image = data_augmentation(image)
                return image, label

            self.train_generator = tf.keras.utils.image_dataset_from_directory(
                directory=self.config.training_data,
                batch_size=self.config.params_batch_size,
                image_size=image_size,
                labels='inferred',
                label_mode='categorical',
                class_names=['adenocarcinoma', 'normal'],
                subset='training',
                seed=seed,
                color_mode='rgb' if self.config.params_image_size[2] == 3 else 'grayscale',
                validation_split=0.2,
            ).map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).cache().prefetch(tf.data.AUTOTUNE)

            logger.info('Train generator created successfully')
        except Exception as e:
            logger.error(f'Failed train generator: {e}')
            self.train_generator = None

        try:
            logger.info('Trying validation generator')
            self.val_generator = tf.keras.utils.image_dataset_from_directory(
                directory=self.config.training_data,
                batch_size=self.config.params_batch_size,
                image_size=image_size,
                labels='inferred',
                label_mode='categorical',
                class_names=['adenocarcinoma', 'normal'],
                subset='validation',
                seed=seed,
                color_mode='rgb' if self.config.params_image_size[2] == 3 else 'grayscale',
                validation_split=0.2,
            ).cache().prefetch(tf.data.AUTOTUNE)

            logger.info('Validation generator created successfully')
        except Exception as e:
            logger.error(f'Failed validation generator: {e}')
            self.val_generator = None

    def train(self):
        if self.model is not None and self.train_generator is not None and self.val_generator is not None:
            self.model.fit(
                self.train_generator,
                validation_data=self.val_generator,
                epochs=self.config.params_epoch
            )
            logger.info("Model training completed.")

            self.save_model(path=self.config.trained_model_path, model=self.model)
        else:
            logger.error("Training or validation generator not initialized properly or model not loaded.")

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

