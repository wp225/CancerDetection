from urllib.parse import urlparse

import mlflow
import tensorflow as tf
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import *
import config

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _valid_generator(self):

        self.valid_generator = tf.keras.utils.image_dataset_from_directory(
            directory=self.config.training_data,
            batch_size=self.config.params_batch_size,
            image_size=tuple(self.config.params_image_size[:2]),
            labels='inferred',
            label_mode='categorical',
            class_names=['adenocarcinoma', 'normal'],
            subset='validation',
            seed=143,
            color_mode='rgb' if self.config.params_image_size[2] == 3 else 'grayscale',
            validation_split=0.2,
        ).cache().prefetch(tf.data.AUTOTUNE)

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    def evaluation(self):
        self.model = self.load_model(self.config.path_to_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )
            # Model registry does not work with file store
            try:
                if tracking_url_type_store != "file":

                    # Register the model
                    # There are other ways to use the Model Registry, which depends on the use case,
                    # please refer to the doc for more information:
                    # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                    mlflow.keras.log_model(self.model, "model", registered_model_name="VGG16Model")
                else:
                    mlflow.keras.log_model(self.model, "model")

                logger.info('Model saved Successfully')
            except Exception as e:
                logger.info('Failed saving the model')

if __name__ == '__main__':
    config = ConfigurationManager()
    eval_config = config.get_evaluation_config()
    evaluation = Evaluation(eval_config)
    evaluation.evaluation()
    evaluation.log_into_mlflow()