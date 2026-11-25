import argparse
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    ALL_FEATURES,
    GOLD_DIR,
    MODEL_ARTIFACT_DIR,
    TARGET_COL,
)

DRIFT_P_VALUE_THRESHOLD = 0.05
PERFORMANCE_DEGRADATION_THRESHOLD_PCT = 15.0  # Retrain if RMSE degrades by > 15%
MAX_ACCEPTABLE_RMSE = 7.0  # minutes


def detect_data_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> dict:
    print("\n--- 1. Statistical Data Drift Audit (Inputs X) ---")
    results = {}

    # 1. Trip Distance Drift (Continuous)
    ref_dist = np.asarray(reference_df["trip_distance"].values, dtype=np.float64)
    cur_dist = np.asarray(current_df["trip_distance"].values, dtype=np.float64)
    ks_stat, ks_p = ks_2samp(ref_dist, cur_dist)
    w_dist = float(wasserstein_distance(ref_dist, cur_dist))
    distance_drift = ks_p < DRIFT_P_VALUE_THRESHOLD

    print(f"Trip Distance KS-test: stat={ks_stat:.4f}, p-value={ks_p:.4e}")
    print(f"Trip Distance Wasserstein shift: {w_dist:.3f} miles")
    print(f"Trip Distance Data Drift: {'DETECTED' if distance_drift else 'NONE'}")
    results["trip_distance"] = {
        "ks_stat": float(ks_stat),
        "p_value": float(ks_p),
        "wasserstein": float(w_dist),
        "drift": distance_drift,
    }

    # 2. Pickup Hour Drift (Temporal)
    hour_stat, hour_p = ks_2samp(reference_df["pickup_hour"], current_df["pickup_hour"])
    hour_drift = hour_p < DRIFT_P_VALUE_THRESHOLD
    print(f"Pickup Hour KS-test: stat={hour_stat:.4f}, p-value={hour_p:.4e}")
    print(f"Pickup Hour Data Drift: {'DETECTED' if hour_drift else 'NONE'}")
    results["pickup_hour"] = {"stat": float(hour_stat), "p_value": float(hour_p), "drift": hour_drift}

    # 3. Categorical Zone Overlap Check
    ref_zones = set(reference_df["PULocationID"].unique())
    cur_zones = set(current_df["PULocationID"].unique())
    new_zones = cur_zones - ref_zones
    print(f"Pickup Zones Analyzed: {len(cur_zones)} (Unseen zones: {len(new_zones)})")
    results["new_zones"] = list(new_zones)

    return results


def detect_model_drift(
    current_df: pd.DataFrame,
    baseline_rmse: float = 6.3940,
) -> tuple[bool, dict]:
    print("\n--- 2. Model & Performance Drift Audit (P(Y|X)) ---")
    model_path = MODEL_ARTIFACT_DIR / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Active model not found at {model_path}")

    model = joblib.load(model_path)
    X = current_df[ALL_FEATURES]
    y_true = current_df[TARGET_COL]

    y_pred = model.predict(X)
    current_rmse = root_mean_squared_error(y_true, y_pred)
    current_mae = mean_absolute_error(y_true, y_pred)
    current_r2 = r2_score(y_true, y_pred)
    mean_residual = float(np.mean(y_pred - y_true))

    pct_degraded = ((current_rmse - baseline_rmse) / baseline_rmse) * 100.0

    print(f"Baseline Benchmark RMSE: {baseline_rmse:.4f} mins")
    print(f"Current Evaluated RMSE:  {current_rmse:.4f} mins")
    print(f"Current MAE:             {current_mae:.4f} mins")
    print(f"Current R2:              {current_r2:.4f}")
    print(f"Mean Residual Bias:      {mean_residual:+.3f} mins")
    print(f"Performance Change:      {pct_degraded:+.2f}%")

    # Smart Trigger Decision
    needs_retrain = (pct_degraded > PERFORMANCE_DEGRADATION_THRESHOLD_PCT) or (current_rmse > MAX_ACCEPTABLE_RMSE)

    print("\n=== SMART RETRAINING DECISION ===")
    if needs_retrain:
        print("🚨 PERFORMANCE DRIFT DETECTED!")
        print(f"Reason: RMSE degraded by {pct_degraded:.1f}% (exceeds {PERFORMANCE_DEGRADATION_THRESHOLD_PCT}% threshold).")
        print("ACTION: Trigger automated retraining pipeline on recent rolling window.")
    else:
        print("✅ MODEL IS HEALTHY & ACCURATE!")
        print(f"Reason: RMSE change ({pct_degraded:+.1f}%) is within tolerance (< {PERFORMANCE_DEGRADATION_THRESHOLD_PCT}%).")
        print("ACTION: Retraining skipped. Saved compute resources and prevented unnecessary model thrashing.")

    metrics = {
        "baseline_rmse": baseline_rmse,
        "current_rmse": current_rmse,
        "pct_degraded": pct_degraded,
        "mean_residual": mean_residual,
        "needs_retrain": needs_retrain,
    }
    return needs_retrain, metrics


def run_drift_pipeline(sample_size: int = 50_000) -> bool:
    train_path = GOLD_DIR / "train.parquet"
    test_path = GOLD_DIR / "frozen_test.parquet"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Reference data not found in data/gold/. Run split.py first.")

    print(f"Loading reference training set ({train_path})...")
    ref_df = pd.read_parquet(train_path).sample(n=sample_size, random_state=42)
    print(f"Loading current evaluation set ({test_path})...")
    cur_df = pd.read_parquet(test_path).sample(n=sample_size, random_state=42)

    detect_data_drift(ref_df, cur_df)
    needs_retrain, _ = detect_model_drift(cur_df)
    return needs_retrain


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit data drift and performance drift")
    parser.add_argument("--sample-size", type=int, default=50_000)
    args = parser.parse_args()

    run_drift_pipeline(args.sample_size)
