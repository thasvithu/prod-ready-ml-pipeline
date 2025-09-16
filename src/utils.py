import os
import sys
import dill
import numpy as np
from typing import Any, Dict

from src.exception import CustomException
from src.logger import logging


def save_object(file_path: str, obj: Any) -> None:
	"""Serialize and save any Python object to disk using dill.

	Args:
		file_path: Target file path where the object will be saved.
		obj: Python object to persist.
	"""
	try:
		dir_path = os.path.dirname(file_path)
		if dir_path:
			os.makedirs(dir_path, exist_ok=True)

		with open(file_path, "wb") as file_obj:
			dill.dump(obj, file_obj)
		logging.info(f"Saved object to {file_path}")
	except Exception as e:
		raise CustomException(e)


def load_object(file_path: str) -> Any:
	"""Load a Python object from disk previously saved with dill.

	Args:
		file_path: Path to the serialized object file.

	Returns:
		Deserialized Python object.
	"""
	try:
		with open(file_path, "rb") as file_obj:
			obj = dill.load(file_obj)
		logging.info(f"Loaded object from {file_path}")
		return obj
	except Exception as e:
		raise CustomException(e)


def evaluate_models(
	X_train: np.ndarray,
	y_train: np.ndarray,
	X_test: np.ndarray,
	y_test: np.ndarray,
	models: Dict[str, Any],
	scorer: Any,
) -> Dict[str, float]:
	"""Train and evaluate multiple models and return their scores.

	Args:
		X_train, y_train, X_test, y_test: Train/test splits.
		models: Mapping from model name to instantiated estimator.
		scorer: A callable scorer with signature scorer(y_true, y_pred) -> float

	Returns:
		Dict of model_name -> score on test set.
	"""
	from sklearn.base import clone

	report: Dict[str, float] = {}
	try:
		for name, model in models.items():
			est = clone(model)
			logging.info(f"Training model: {name}")
			est.fit(X_train, y_train)
			y_pred = est.predict(X_test)
			score = float(scorer(y_test, y_pred))
			logging.info(f"Model: {name} Score: {score:.4f}")
			report[name] = score
		return report
	except Exception as e:
		raise CustomException(e)

