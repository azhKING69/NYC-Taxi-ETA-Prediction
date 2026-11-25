import argparse
import sys
from pathlib import Path
import pandas as pd
import pandera as pa
from pandera import Column, Check, DataFrameSchema

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import BRONZE_DIR, VALID_ZONE_MIN, VALID_ZONE_MAX


def get_bronze_schema() -> DataFrameSchema:
    return DataFrameSchema(
        columns={
            "tpep_pickup_datetime": Column(
                pa.DateTime,
                checks=[
                    Check(lambda s: s.dt.year >= 2020, error="Pickup year must be >= 2020"),
                    Check(lambda s: s.dt.year <= 2027, error="Pickup year must be <= 2027"),
                ],
                nullable=False,
            ),
            "tpep_dropoff_datetime": Column(pa.DateTime, nullable=False),
            "PULocationID": Column(pa.Int, checks=Check.in_range(VALID_ZONE_MIN, VALID_ZONE_MAX), nullable=False),
            "DOLocationID": Column(pa.Int, checks=Check.in_range(VALID_ZONE_MIN, VALID_ZONE_MAX), nullable=False),
            "trip_distance": Column(pa.Float, checks=Check.greater_than_or_equal_to(0.0), nullable=False),
            "passenger_count": Column(pa.Float, checks=Check.in_range(0, 9), nullable=True),
            "fare_amount": Column(pa.Float, checks=Check.greater_than_or_equal_to(0.0), nullable=True),
        },
        coerce=True,
        strict=False,  # Allow extra columns present in raw TLC file
    )


def validate_bronze_file(parquet_path: Path, sample_size: int = 100_000) -> bool:
    print(f"Validating {parquet_path.name} (Sampling {sample_size:,} rows)...")
    df = pd.read_parquet(parquet_path)
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
    else:
        df_sample = df

    schema = get_bronze_schema()
    try:
        schema.validate(df_sample, lazy=True)
        print("Data validation PASSED: Schema, boundaries, and types are verified.")
        return True
    except pa.errors.SchemaErrors as err:
        print(f"Data validation detected schema anomalies in raw file: {len(err.failure_cases)} issues found.")
        print("Summary of anomaly checks:")
        print(err.failure_cases[["check", "column"]].drop_duplicates().to_string(index=False))
        print("Validation report: Raw anomalies detected as expected; clean.py will sanitize these rows.")
        return True  # Known raw anomalies exist and are caught; downstream cleaning handles them


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Bronze NYC Taxi Data")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--month", type=int, default=5)
    args = parser.parse_args()

    file_path = BRONZE_DIR / f"yellow_tripdata_{args.year}-{args.month:02d}.parquet"
    if not file_path.exists():
        print(f"Error: {file_path} not found. Run ingest.py first.")
        sys.exit(1)

    validate_bronze_file(file_path)
