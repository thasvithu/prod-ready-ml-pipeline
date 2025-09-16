import os
from dataclasses import dataclass, field
from typing import Dict

import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
try:
	from xgboost import XGBRegressor  # type: ignore
except Exception:  # pragma: no cover - optional dependency
	XGBRegressor = None  # type: ignore

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_object


@dataclass
class ModelTrainerConfig:
	artifacts_dir: str = field(default_factory=lambda: os.getenv("ARTIFACTS_DIR", "artifacts"))
	trained_model_file_path: str = field(init=False)
	metric_name: str = "r2"
	min_improvement: float = 0.0

	def __post_init__(self):
		self.trained_model_file_path = os.path.join(self.artifacts_dir, "model.pkl")


class ModelTrainer:
	def __init__(self, config: ModelTrainerConfig = None):
		self.config = config or ModelTrainerConfig()

	def initiate_model_trainer(
		self,
		X_train: np.ndarray,
		X_test: np.ndarray,
		y_train: np.ndarray,
		y_test: np.ndarray,
	) -> Dict[str, float]:
		try:
			logging.info("Starting model training")

			models = {
				"LinearRegression": LinearRegression(),
				"RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
			}
			if XGBRegressor is not None:
				models["XGBRegressor"] = XGBRegressor(
					n_estimators=300,
					learning_rate=0.05,
					max_depth=4,
					subsample=0.9,
					colsample_bytree=0.9,
					random_state=42,
					objective="reg:squarederror",
				)
			else:
				logging.warning("XGBoost is not installed; skipping XGBRegressor")

			def scorer(y_true, y_pred):
				return r2_score(y_true, y_pred)

			report = evaluate_models(X_train, y_train, X_test, y_test, models, scorer)

			# Select best model
			best_model_name = max(report, key=report.get)
			best_model_score = report[best_model_name]
			logging.info(f"Best model: {best_model_name} with score {best_model_score:.4f}")

			# Refit the best model on full train and persist
			best_model = models[best_model_name]
			best_model.fit(X_train, y_train)

			os.makedirs(os.path.dirname(self.config.trained_model_file_path), exist_ok=True)
			save_object(self.config.trained_model_file_path, best_model)
			logging.info(f"Saved model to {self.config.trained_model_file_path}")

			report["best_model"] = best_model_name
			report["best_score"] = float(best_model_score)
			report["model_path"] = self.config.trained_model_file_path
			return report
		except Exception as e:
			raise CustomException(e)

