from cnnClassifier.components.prepare_base_model import PrepareBaseModel
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier import logger

STAGE_NAME = "Prepare base model"

class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):  # Tambahkan 'self' sebagai parameter
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.get_base_model()
        prepare_base_model.update_base_model()

if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()  # Panggil metode main pada instance obj
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(f"Error has occurred in {STAGE_NAME}: {e}")
        raise e