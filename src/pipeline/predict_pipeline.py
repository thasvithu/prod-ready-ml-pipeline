import os
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Union

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


@dataclass
class PredictConfig:
	artifacts_dir: str = field(default_factory=lambda: os.getenv("ARTIFACTS_DIR", "artifacts"))
	preprocessor_path: str = field(init=False)
	model_path: str = field(init=False)

	def __post_init__(self):
		self.preprocessor_path = os.path.join(self.artifacts_dir, "preprocessor.pkl")
		self.model_path = os.path.join(self.artifacts_dir, "model.pkl")


class PredictPipeline:
	def __init__(self, config: PredictConfig = None):
		self.config = config or PredictConfig()
		self._preprocessor = None
		self._model = None

	def _load_artifacts(self):
		if self._preprocessor is None:
			self._preprocessor = load_object(self.config.preprocessor_path)
		if self._model is None:
			self._model = load_object(self.config.model_path)

	def predict(self, data: Union[pd.DataFrame, List[dict]]):
		try:
			self._load_artifacts()
			if isinstance(data, list):
				df = pd.DataFrame(data)
			else:
				df = data
			logging.info(f"Predicting on input shape: {df.shape}")
			X = self._preprocessor.transform(df)
			preds = self._model.predict(X)
			return preds
		except Exception as e:
			raise CustomException(e)

