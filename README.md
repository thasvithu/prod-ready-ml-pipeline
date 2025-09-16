# Prod-Ready ML Pipeline

An end-to-end, production-ready machine learning pipeline for predicting student performance (math score) using a reproducible structure with the following stages:

- Data ingestion from CSV into train/test artifacts
- Data preprocessing with scikit-learn pipelines (imputation, scaling, one-hot encoding)
- Model training and selection across multiple regressors (Linear Regression, RandomForest, XGBoost)
- Model and preprocessor persistence
- Prediction pipeline for serving
- Basic unit tests and an integration test

## Project Structure

- `src/components`: Ingestion, transformation, and model training modules
- `src/pipeline`: Train and predict entry points
- `notebooks/data`: Dataset (`stud.csv`)
- `artifacts`: Created at runtime for raw/train/test CSVs, preprocessor, and trained model
- `tests`: Minimal tests for utilities and training pipeline

## Setup

1) Create and activate a Python 3.9+ virtual environment
2) Install requirements
3) Run training
4) Run prediction (example)

Commands (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.pipeline.train_pipeline

# Example prediction after training
python - <<'PY'
from src.pipeline.predict_pipeline import PredictPipeline
import pandas as pd

data = pd.read_csv('notebooks/data/stud.csv').drop(columns=['math_score']).head(5)
preds = PredictPipeline().predict(data)
print(preds)
PY
```

## Tests

```powershell
pip install pytest
pytest -q
```

## Notes

- Default target is `math_score`. You can adapt the transformation to predict other targets.
- Artifacts are saved under `artifacts/` by default.
