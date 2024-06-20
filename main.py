from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelPipeline
from cnnClassifier.pipeline.stage_03_model_trainer import ModelTrainerPipeline
from cnnClassifier.pipeline.stage_04_model_evaluation import EvaluationPipeline
from cnnClassifier import logger

def main():

    stage_name = 'Data Ingestion'
    logger.info(f'------Starting {stage_name}-------')
    try:
        obj = DataIngestionTrainingPipeline()
        obj.main()
    except:
        logger.info(f'------Failed {stage_name}-------')



    stage_name = 'Model Preperation'
    logger.info(f'------Starting {stage_name}-------')
    try:
        obj = PrepareBaseModelPipeline()
        obj.main()
    except:
        logger.info(f'------Failed {stage_name}-------')



    stage_name = 'Model Training'
    logger.info(f'------Starting {stage_name}-------')
    try:
        obj = ModelTrainerPipeline()
        obj.main()
    except:
        logger.info(f'------Failed {stage_name}-------')



    stage_name = 'Model Evaluation'
    logger.info(f'------Starting {stage_name}-------')
    try:
        obj = EvaluationPipeline()
        obj.main()
    except:
        logger.info(f'------Failed {stage_name}-------')




if __name__ == '__main__':
    main()