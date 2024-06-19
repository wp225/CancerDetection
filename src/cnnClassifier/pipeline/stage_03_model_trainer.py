from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.entity.config_entity import TrainingConfig
from cnnClassifier.components.model_trainer import Train
import config
class ModelTrainerPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        Train_config = config.get_training_config()
        trainer = Train(Train_config)
        trainer.get_base_model()
        trainer.train_valid_generator()
        trainer.train()

if __name__ == '__main__':
    obj = ModelTrainerPipeline()
    obj.main()