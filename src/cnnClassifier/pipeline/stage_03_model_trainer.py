from datetime import datetime

import mlflow
# import config

from cnnClassifier import logger
from cnnClassifier.components.model_trainer import Train
from cnnClassifier.config.configuration import ConfigurationManager


class ModelTrainerPipeline:
    def __init__(self):
        pass

    def main(self):
        exp_name = datetime.now().date()
        run_name = datetime.now().strftime('%H:%M')
        mlflow.set_experiment(experiment_name = str(exp_name))
        config = ConfigurationManager()
        Train_config = config.get_training_config()
        trainer = Train(Train_config)
        trainer.get_base_model()
        trainer.train_valid_generator()
        with mlflow.start_run(run_name=str(run_name)):
            logger.info('Starting Model Training')
            trainer.train()


if __name__ == '__main__':
    obj = ModelTrainerPipeline()
    obj.main()
