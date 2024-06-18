from cnnClassifier import logger
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.data_ingestion import DataIngestion

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        STAGE_NAME = 'Data Ingestion'
        logger.info(f"--Starting Stage {STAGE_NAME}--")
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()

if __name__ == '__main__':
    obj = DataIngestionTrainingPipeline()
    obj.main()