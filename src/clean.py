import argparse
import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    BRONZE_DIR,
    SILVER_DIR,
    MIN_TRIP_DISTANCE,
    MAX_TRIP_DISTANCE,
    MIN_TRIP_DURATION,
    MAX_TRIP_DURATION,
    MIN_FARE_AMOUNT,
    MAX_FARE_AMOUNT,
    VALID_ZONE_MIN,
    VALID_ZONE_MAX,
    TARGET_COL,
)


def clean_data(year: int, month: int, sample_limit: int | None = None) -> Path:
    raw_file = BRONZE_DIR / f"yellow_tripdata_{year}-{month:02d}.parquet"
    if not raw_file.exists():
        raise FileNotFoundError(f"Bronze file {raw_file} not found. Run ingest.py first.")

    SILVER_DIR.mkdir(parents=True, exist_ok=True)
    out_file = SILVER_DIR / f"cleaned_{year}-{month:02d}.parquet"

    print(f"Loading {raw_file.name}...")
    df = pd.read_parquet(raw_file)
    initial_rows = len(df)
    print(f"Initial raw rows: {initial_rows:,}")

    if sample_limit and sample_limit < initial_rows:
        df = df.sample(n=sample_limit, random_state=42)
        print(f"Subsampled to {len(df):,} rows for processing.")

    # 1. Filter timestamps strictly to the target month & year
    df = df[
        (df["tpep_pickup_datetime"].dt.year == year)
        & (df["tpep_pickup_datetime"].dt.month == month)
        & df["tpep_pickup_datetime"].notna()
        & df["tpep_dropoff_datetime"].notna()
    ]

    # 2. Derive target variable: trip_duration_minutes
    duration = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60.0
    df[TARGET_COL] = duration

    # 3. Filter durations (1 to 120 minutes)
    df = df[(df[TARGET_COL] >= MIN_TRIP_DURATION) & (df[TARGET_COL] <= MAX_TRIP_DURATION)]

    # 4. Filter distance (0.1 to 100 miles)
    df = df[(df["trip_distance"] >= MIN_TRIP_DISTANCE) & (df["trip_distance"] <= MAX_TRIP_DISTANCE)]

    # 5. Filter fare amounts
    if "fare_amount" in df.columns:
        df = df[(df["fare_amount"] >= MIN_FARE_AMOUNT) & (df["fare_amount"] <= MAX_FARE_AMOUNT)]

    # 6. Filter zone IDs (1 to 265)
    df = df[
        (df["PULocationID"] >= VALID_ZONE_MIN)
        & (df["PULocationID"] <= VALID_ZONE_MAX)
        & (df["DOLocationID"] >= VALID_ZONE_MIN)
        & (df["DOLocationID"] <= VALID_ZONE_MAX)
    ]

    # 7. Impute missing passenger_count with mode (1.0)
    df["passenger_count"] = df["passenger_count"].fillna(1.0).astype(int)

    cleaned_rows = len(df)
    drop_pct = (initial_rows - cleaned_rows) / initial_rows * 100
    print(f"Cleaned data rows: {cleaned_rows:,} (Removed {drop_pct:.2f}% noisy/invalid records)")

    df.to_parquet(out_file, index=False)
    print(f"Saved cleaned data to {out_file}")
    return out_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean raw bronze taxi data")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--month", type=int, default=5)
    parser.add_argument("--sample-limit", type=int, default=None, help="Optional row limit")
    args = parser.parse_args()

    clean_data(args.year, args.month, args.sample_limit)
