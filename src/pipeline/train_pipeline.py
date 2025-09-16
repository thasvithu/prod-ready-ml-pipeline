from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.data_transformation import (
	DataTransformation,
	DataTransformationConfig,
)
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig
from src.exception import CustomException
from src.logger import logging


def run_training_pipeline():
	try:
		logging.info("=== Training pipeline started ===")
		# Ingestion
		ingestion = DataIngestion(DataIngestionConfig())
		train_path, test_path = ingestion.initiate_data_ingestion()

		# Transformation
		transformation = DataTransformation(DataTransformationConfig())
		X_train, X_test, y_train, y_test, preproc_path = transformation.initiate_data_transformation(
			train_path, test_path
		)

		# Train model
		trainer = ModelTrainer(ModelTrainerConfig())
		report = trainer.initiate_model_trainer(X_train, X_test, y_train, y_test)
		report["preprocessor_path"] = preproc_path
		logging.info(f"Training completed. Summary: {report}")
		logging.info("=== Training pipeline completed ===")
		return report
	except Exception as e:
		raise CustomException(e)


if __name__ == "__main__":
	run_training_pipeline()

