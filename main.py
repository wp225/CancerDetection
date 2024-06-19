from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelPipeline
from cnnClassifier.pipeline.stage_03_model_trainer import ModelTrainerPipeline

obj = DataIngestionTrainingPipeline()
obj.main()

obj = PrepareBaseModelPipeline()
obj.main()

obj = ModelTrainerPipeline()
obj.main()

