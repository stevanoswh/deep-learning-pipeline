import tensorflow as tf
from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.stage_02_prepare_model_base import PrepareBaseModelTrainingPipeline
from cnnClassifier.pipeline.stage_03_training import TrainingPipeline
from cnnClassifier.pipeline.stage_04_evaluation import EvaluationPipeline

def configure_tensorflow_for_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Atur agar TensorFlow hanya menggunakan GPU yang tersedia
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Detected {len(gpus)} GPU(s): {gpus}")
        except RuntimeError as e:
            # Kesalahan terjadi jika runtime sudah dimulai
            logger.exception(f"Error setting up GPU: {e}")
    else:
        logger.info("No GPUs detected, using CPU.")

def run_pipeline_stage(stage_name, pipeline_class):
    try:
        logger.info(f"=====================")
        logger.info(f">>>>>> Stage {stage_name} started <<<<<<")
        pipeline_obj = pipeline_class()
        pipeline_obj.main()
        logger.info(f">>>>>> Stage {stage_name} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(f"Error occurred in stage {stage_name}: {e}")
        raise e

def main():
    # Konfigurasi TensorFlow untuk menggunakan GPU
    configure_tensorflow_for_gpu()

    # Data Ingestion Stage
    run_pipeline_stage("Data Ingestion Stage", DataIngestionTrainingPipeline)

    # Prepare Base Model Stage
    run_pipeline_stage("Prepare Base Model", PrepareBaseModelTrainingPipeline)

    # Training Stage
    run_pipeline_stage("Training", TrainingPipeline)

    # Evaluation Stage
    run_pipeline_stage("Evaluation", EvaluationPipeline)

if __name__ == "__main__":
    main()
 