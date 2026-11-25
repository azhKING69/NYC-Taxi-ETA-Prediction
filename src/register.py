import argparse
import sys
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import MLFLOW_TRACKING_URI, REGISTERED_MODEL_NAME


def register_production_model(run_id: str, stage: str = "Production") -> str:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    model_uri = f"runs:/{run_id}/model"
    print(f"Registering model from run {run_id} as '{REGISTERED_MODEL_NAME}'...")

    # Create registered model if not exists
    try:
        client.create_registered_model(REGISTERED_MODEL_NAME)
    except Exception:
        pass  # Model already exists

    # Create model version
    model_version = client.create_model_version(
        name=REGISTERED_MODEL_NAME,
        source=model_uri,
        run_id=run_id,
        description="Automated training run for NYC taxi duration predictor",
    )
    version = model_version.version
    print(f"Created model version {version}")

    # Set alias champion / tag
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", version)
    client.set_model_version_tag(REGISTERED_MODEL_NAME, version, "stage", stage)
    print(f"Model version {version} promoted with alias 'champion' (Stage: {stage})")
    return version


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register winning model in MLflow")
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--stage", type=str, default="Production")
    args = parser.parse_args()

    register_production_model(args.run_id, args.stage)
