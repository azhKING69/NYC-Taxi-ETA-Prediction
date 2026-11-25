import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.ingest import download_monthly_data
from src.validate import validate_bronze_file
from src.clean import clean_data
from src.features import extract_features
from src.split import split_data
from src.train import run_training
from src.evaluate import evaluate_model
from src.register import register_production_model
from src.export_model import export_model_artifact


def execute_full_pipeline(year: int = 2026, month: int = 5, sample_limit: int = 500_000) -> str:
    print(f"\n==================================================================")
    print(f"🚀 EXECUTING NYC TAXI MLOPS PIPELINE: YEAR {year}, MONTH {month:02d}")
    print(f"==================================================================")

    # 1. Ingest
    print("\n[Stage 1/7] Data Ingestion (Bronze Layer)...")
    bronze_path = download_monthly_data(year, month)

    # 2. Validate
    print("\n[Stage 2/7] Data Schema & Boundary Validation (Pandera)...")
    validate_bronze_file(bronze_path)

    # 3. Clean
    print("\n[Stage 3/7] Data Cleaning & Label Derivation (Silver Layer)...")
    clean_data(year, month)

    # 4. Feature Engineering
    print("\n[Stage 4/7] Spatio-Temporal Feature Engineering (Gold Layer)...")
    extract_features(year, month)

    # 5. Split
    print("\n[Stage 5/7] Train / Val / Frozen Test Benchmark Split...")
    split_data(year, month, sample_limit=sample_limit)

    # 6. Train with MLflow Tracking
    print("\n[Stage 6/7] LightGBM Multi-Config Training & MLflow Logging...")
    best_run_id, best_val_rmse = run_training(data_version=f"{year}-{month:02d}")

    # 7. Evaluate, Register, Export
    print("\n[Stage 7/7] Evaluation Gate, Model Registry Promotion & Export...")
    passed, test_rmse = evaluate_model(best_run_id)
    if not passed:
        print("Pipeline aborted: Candidate model did not pass the evaluation gate.")
        sys.exit(1)

    version = register_production_model(best_run_id, stage="Production")
    export_path = export_model_artifact(best_run_id)

    print(f"\n==================================================================")
    print(f"🎉 PIPELINE EXECUTION COMPLETE!")
    print(f"Production Model Version: {version}")
    print(f"Champion Artifact Exported: {export_path}")
    print(f"Benchmark Test RMSE: {test_rmse:.4f} minutes")
    print(f"==================================================================\n")
    return best_run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute full NYC Taxi MLOps pipeline")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--month", type=int, default=5)
    parser.add_argument("--sample-limit", type=int, default=500_000)
    args = parser.parse_args()

    execute_full_pipeline(args.year, args.month, args.sample_limit)
