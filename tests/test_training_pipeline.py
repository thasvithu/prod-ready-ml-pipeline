import os
from src.pipeline.train_pipeline import run_training_pipeline


def test_training_pipeline_creates_artifacts(tmp_path, monkeypatch):
    # Redirect artifacts directory to temp to avoid polluting repo
    artifacts_dir = tmp_path / "artifacts"
    (artifacts_dir).mkdir(parents=True, exist_ok=True)

    # Set environment variable so all configs write into tmp artifacts
    monkeypatch.setenv("ARTIFACTS_DIR", str(artifacts_dir))

    report = run_training_pipeline()
    assert os.path.exists(report["model_path"])  # model created
    assert os.path.exists(report["preprocessor_path"])  # preprocessor created
