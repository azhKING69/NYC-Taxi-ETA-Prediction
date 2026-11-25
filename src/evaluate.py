import argparse
import sys
from pathlib import Path
import mlflow
import mlflow.lightgbm
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import ALL_FEATURES, GOLD_DIR, MLFLOW_TRACKING_URI, TARGET_COL

RMSE_GATE_THRESHOLD = 7.0  # realistic urban ETA threshold (standard baseline error is ~15 mins)


def evaluate_model(run_id: str, threshold: float = RMSE_GATE_THRESHOLD) -> tuple[bool, float]:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    frozen_test_path = GOLD_DIR / "frozen_test.parquet"

    if not frozen_test_path.exists():
        raise FileNotFoundError(f"Frozen test set not found at {frozen_test_path}")

    print(f"Loading frozen test benchmark from {frozen_test_path}...")
    test_df = pd.read_parquet(frozen_test_path)
    X_test = test_df[ALL_FEATURES]
    y_test = test_df[TARGET_COL]

    model_uri = f"runs:/{run_id}/model"
    print(f"Loading model from {model_uri}...")
    model = mlflow.lightgbm.load_model(model_uri)

    preds = model.predict(X_test)
    test_rmse = root_mean_squared_error(y_test, preds)
    test_mae = mean_absolute_error(y_test, preds)
    test_r2 = r2_score(y_test, preds)

    print("\n=== Frozen Benchmark Evaluation Results ===")
    print(f"Test RMSE: {test_rmse:.4f} mins (Gate Threshold: < {threshold:.1f} mins)")
    print(f"Test MAE:  {test_mae:.4f} mins")
    print(f"Test R2:   {test_r2:.4f}")

    # Log metrics to MLflow run
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({
            "frozen_test_rmse": test_rmse,
            "frozen_test_mae": test_mae,
            "frozen_test_r2": test_r2,
        })

    passed = test_rmse < threshold
    if passed:
        print("GATE CHECK: PASSED! Model meets accuracy criteria for production promotion.")
    else:
        print(f"GATE CHECK: FAILED! Model RMSE ({test_rmse:.4f}) exceeds threshold ({threshold}).")

    return passed, test_rmse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model against frozen benchmark")
    parser.add_argument("--run-id", type=str, required=True, help="MLflow run ID to evaluate")
    parser.add_argument("--threshold", type=float, default=RMSE_GATE_THRESHOLD, help="RMSE gate threshold")
    args = parser.parse_args()

    passed, _ = evaluate_model(args.run_id, args.threshold)
    if not passed:
        sys.exit(1)
