import argparse
import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    SILVER_DIR,
    GOLD_DIR,
    AIRPORT_ZONES,
    ALL_FEATURES,
    TARGET_COL,
)


def extract_features(year: int, month: int) -> Path:
    silver_file = SILVER_DIR / f"cleaned_{year}-{month:02d}.parquet"
    if not silver_file.exists():
        raise FileNotFoundError(f"Cleaned file {silver_file} not found. Run clean.py first.")

    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    out_file = GOLD_DIR / f"featured_{year}-{month:02d}.parquet"

    print(f"Extracting features from {silver_file.name}...")
    df = pd.read_parquet(silver_file)

    # Temporal feature engineering
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour.astype(int)
    df["pickup_day_of_week"] = df["tpep_pickup_datetime"].dt.dayofweek.astype(int)
    df["is_weekend"] = (df["pickup_day_of_week"] >= 5).astype(int)

    # Domain indicators: rush hour (7-10 AM, 4-7 PM on weekdays)
    is_weekday = df["pickup_day_of_week"] < 5
    is_am_rush = df["pickup_hour"].between(7, 10)
    is_pm_rush = df["pickup_hour"].between(16, 19)
    df["is_rush_hour"] = (is_weekday & (is_am_rush | is_pm_rush)).astype(int)

    # Spatial indicators: airport trip (JFK=132, LGA=138, Newark=1)
    df["is_airport_trip"] = (
        df["PULocationID"].isin(AIRPORT_ZONES) | df["DOLocationID"].isin(AIRPORT_ZONES)
    ).astype(int)

    # Retain strictly inference features + target
    output_cols = ALL_FEATURES + [TARGET_COL]
    gold_df = df[output_cols].copy()

    # Ensure integer types for categorical locations
    gold_df["PULocationID"] = gold_df["PULocationID"].astype(int)
    gold_df["DOLocationID"] = gold_df["DOLocationID"].astype(int)

    print(f"Features created: {ALL_FEATURES}")
    print(f"Featured dataset shape: {gold_df.shape}")

    gold_df.to_parquet(out_file, index=False)
    print(f"Saved featured data to {out_file}")
    return out_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Engineer features for modeling")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--month", type=int, default=5)
    args = parser.parse_args()

    extract_features(args.year, args.month)
