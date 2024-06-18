from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelPipeline

obj = DataIngestionTrainingPipeline()
obj.main()

obj = PrepareBaseModelPipeline()
obj.main()
