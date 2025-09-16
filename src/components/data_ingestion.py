import os
import pandas as pd
from dataclasses import dataclass, field
from sklearn.model_selection import train_test_split

from src.logger import logging
from src.exception import CustomException


@dataclass
class DataIngestionConfig:
	artifacts_dir: str = field(default_factory=lambda: os.getenv("ARTIFACTS_DIR", "artifacts"))
	raw_data_path: str = field(init=False)
	train_data_path: str = field(init=False)
	test_data_path: str = field(init=False)
	source_data_path: str = field(default_factory=lambda: os.path.join("notebooks", "data", "stud.csv"))

	def __post_init__(self):
		self.raw_data_path = os.path.join(self.artifacts_dir, "raw.csv")
		self.train_data_path = os.path.join(self.artifacts_dir, "train.csv")
		self.test_data_path = os.path.join(self.artifacts_dir, "test.csv")


class DataIngestion:
	def __init__(self, config: DataIngestionConfig = None):
		self.config = config or DataIngestionConfig()

	def initiate_data_ingestion(self):
		"""Read the dataset, split into train and test, and save artifacts.

		Returns:
			Tuple of (train_path, test_path)
		"""
		try:
			logging.info("Starting data ingestion")
			df = pd.read_csv(self.config.source_data_path)
			logging.info(f"Loaded data shape: {df.shape}")

			os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)
			df.to_csv(self.config.raw_data_path, index=False)
			logging.info(f"Saved raw data to {self.config.raw_data_path}")

			train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
			train_df.to_csv(self.config.train_data_path, index=False)
			test_df.to_csv(self.config.test_data_path, index=False)
			logging.info(
				f"Saved train ({train_df.shape}) and test ({test_df.shape}) to artifacts"
			)

			return self.config.train_data_path, self.config.test_data_path
		except Exception as e:
			raise CustomException(e)

# Reading the data