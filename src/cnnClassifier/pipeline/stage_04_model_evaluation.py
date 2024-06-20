from cnnClassifier.components.model_evaluation import Evaluation
from cnnClassifier.config.configuration import ConfigurationManager
class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        eval = Evaluation(eval_config)
        eval.evaluation()
        eval.log_into_mlflow()
if __name__ == '__main__':
    obj = EvaluationPipeline()
    obj.main()
