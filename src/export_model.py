import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import joblib
import mlflow
import mlflow.lightgbm

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    ALL_FEATURES,
    MODEL_ARTIFACT_DIR,
    MLFLOW_TRACKING_URI,
    REGISTERED_MODEL_NAME,
)


def export_model_artifact(run_id: str | None = None) -> Path:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    MODEL_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    out_model_path = MODEL_ARTIFACT_DIR / "model.pkl"
    out_meta_path = MODEL_ARTIFACT_DIR / "metadata.json"

    if run_id:
        model_uri = f"runs:/{run_id}/model"
    else:
        model_uri = f"models:/{REGISTERED_MODEL_NAME}@champion"

    print(f"Loading model for export from {model_uri}...")
    model = mlflow.lightgbm.load_model(model_uri)

    # Save via joblib
    joblib.dump(model, out_model_path)
    print(f"Saved model artifact to {out_model_path} ({out_model_path.stat().st_size / 1024:.1f} KB)")

    metadata = {
        "model_name": REGISTERED_MODEL_NAME,
        "run_id": run_id,
        "features": ALL_FEATURES,
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "framework": "lightgbm",
    }
    with open(out_meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved model metadata to {out_meta_path}")

    return out_model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export winning model to model_artifact/")
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()

    export_model_artifact(args.run_id)
