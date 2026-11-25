import argparse
import sys
from pathlib import Path
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    GOLD_DIR,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    TARGET_COL,
)


def run_training(data_version: str = "2026-05") -> tuple[str, float]:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    train_path = GOLD_DIR / "train.parquet"
    val_path = GOLD_DIR / "val.parquet"

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError("Training and validation data not found in data/gold/. Run split.py first.")

    print(f"Loading training data ({train_path}) and validation data ({val_path})...")
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    X_train, y_train = train_df[ALL_FEATURES], train_df[TARGET_COL]
    X_val, y_val = val_df[ALL_FEATURES], val_df[TARGET_COL]

    print(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")

    configs = [
        {"n_estimators": 100, "learning_rate": 0.1, "num_leaves": 31, "max_depth": 6},
        {"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 63, "max_depth": 8},
        {"n_estimators": 300, "learning_rate": 0.03, "num_leaves": 127, "max_depth": 10},
    ]

    best_run_id = None
    best_rmse = float("inf")

    for idx, params in enumerate(configs):
        run_name = f"lgbm_config_{idx}_leaves_{params['num_leaves']}_lr_{params['learning_rate']}"
        print(f"\n--- Running Training [{idx+1}/{len(configs)}]: {run_name} ---")

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_param("data_version", data_version)
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_params(params)

            model = lgb.LGBMRegressor(**params, random_state=42, n_jobs=-1, verbose=-1)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
                categorical_feature=CATEGORICAL_FEATURES,
            )

            preds = model.predict(X_val)
            val_rmse = root_mean_squared_error(y_val, preds)
            val_mae = mean_absolute_error(y_val, preds)
            val_r2 = r2_score(y_val, preds)

            print(f"Validation Results: RMSE = {val_rmse:.4f} mins, MAE = {val_mae:.4f} mins, R2 = {val_r2:.4f}")

            mlflow.log_metrics({
                "val_rmse": val_rmse,
                "val_mae": val_mae,
                "val_r2": val_r2,
            })

            mlflow.lightgbm.log_model(
                lgb_model=model,
                artifact_path="model",
                registered_model_name=None,  # Handled explicitly by register.py
            )

            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_run_id = run.info.run_id

    print(f"\nBest Run ID: {best_run_id} with Val RMSE: {best_rmse:.4f} mins")
    return best_run_id, best_rmse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LightGBM models with MLflow")
    parser.add_argument("--data-version", type=str, default="2026-05")
    args = parser.parse_args()

    run_training(args.data_version)
