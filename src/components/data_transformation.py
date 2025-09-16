import os
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
	artifacts_dir: str = field(default_factory=lambda: os.getenv("ARTIFACTS_DIR", "artifacts"))
	preprocessor_obj_file_path: str = field(init=False)

	def __post_init__(self):
		self.preprocessor_obj_file_path = os.path.join(self.artifacts_dir, "preprocessor.pkl")


class DataTransformation:
	def __init__(self, config: DataTransformationConfig = None):
		self.config = config or DataTransformationConfig()

	def get_preprocessor(self):
		try:
			# Note: Target 'math_score' is excluded from features
			numerical_cols = ["reading_score", "writing_score"]
			categorical_cols = [
				"gender",
				"race_ethnicity",
				"parental_level_of_education",
				"lunch",
				"test_preparation_course",
			]

			num_pipeline = Pipeline(
				steps=[
					("imputer", SimpleImputer(strategy="median")),
					("scaler", StandardScaler()),
				]
			)

			# Ensure dense output for compatibility with models that don't accept sparse matrices
			try:
				onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # sklearn >=1.2
			except TypeError:  # pragma: no cover - for older sklearn
				onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)

			cat_pipeline = Pipeline(
				steps=[
					("imputer", SimpleImputer(strategy="most_frequent")),
					("onehot", onehot),
				]
			)

			preprocessor = ColumnTransformer(
				transformers=[
					("num", num_pipeline, numerical_cols),
					("cat", cat_pipeline, categorical_cols),
				],
				sparse_threshold=0.0,  # force dense output
			)
			return preprocessor
		except Exception as e:
			raise CustomException(e)

	def initiate_data_transformation(self, train_path: str, test_path: str):
		try:
			logging.info("Starting data transformation")
			train_df = pd.read_csv(train_path)
			test_df = pd.read_csv(test_path)

			target_column = "math_score"  # We'll predict math score as example

			X_train = train_df.drop(columns=[target_column])
			y_train = pd.to_numeric(train_df[target_column], errors="coerce")

			X_test = test_df.drop(columns=[target_column])
			y_test = pd.to_numeric(test_df[target_column], errors="coerce")

			# Coerce numeric feature columns in case CSV stored numbers as strings
			for col in ["reading_score", "writing_score"]:
				if col in X_train.columns:
					X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
				if col in X_test.columns:
					X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

			preprocessor = self.get_preprocessor()
			X_train_processed = preprocessor.fit_transform(X_train)
			X_test_processed = preprocessor.transform(X_test)

			os.makedirs(os.path.dirname(self.config.preprocessor_obj_file_path), exist_ok=True)
			save_object(self.config.preprocessor_obj_file_path, preprocessor)
			logging.info(f"Saved preprocessor to {self.config.preprocessor_obj_file_path}")

			return (
				np.array(X_train_processed),
				np.array(X_test_processed),
				np.array(y_train),
				np.array(y_test),
				self.config.preprocessor_obj_file_path,
			)
		except Exception as e:
			raise CustomException(e)

