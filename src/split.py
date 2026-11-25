import argparse
import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import GOLD_DIR


def split_data(
    year: int = 2026,
    month: int = 5,
    test_size: float = 0.10,
    val_size: float = 0.10,
    sample_limit: int | None = 500_000,
) -> tuple[Path, Path, Path]:
    featured_file = GOLD_DIR / f"featured_{year}-{month:02d}.parquet"
    if not featured_file.exists():
        raise FileNotFoundError(f"Featured file {featured_file} not found. Run features.py first.")

    train_path = GOLD_DIR / "train.parquet"
    val_path = GOLD_DIR / "val.parquet"
    frozen_test_path = GOLD_DIR / "frozen_test.parquet"

    print(f"Loading {featured_file.name} for train/val/test split...")
    df = pd.read_parquet(featured_file)

    if sample_limit and len(df) > sample_limit:
        print(f"Sampling {sample_limit:,} rows for rapid multi-config training & evaluation...")
        df = df.sample(n=sample_limit, random_state=42)

    # 1. Carve out frozen test set (10%)
    train_val_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

    # 2. Carve out validation set from remaining (10% of total)
    adjusted_val_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(train_val_df, test_size=adjusted_val_size, random_state=42)

    print(f"Train set: {len(train_df):,} rows")
    print(f"Val set:   {len(val_df):,} rows")
    print(f"Test set:  {len(test_df):,} rows (Frozen benchmark)")

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    # Only create frozen_test if it does not already exist, preserving baseline comparability
    if not frozen_test_path.exists():
        test_df.to_parquet(frozen_test_path, index=False)
        print(f"Created and saved frozen test set to {frozen_test_path}")
    else:
        print(f"Preserving existing frozen test set at {frozen_test_path}")

    return train_path, val_path, frozen_test_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split featured gold data")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--month", type=int, default=5)
    parser.add_argument("--sample-limit", type=int, default=500_000)
    args = parser.parse_args()

    split_data(args.year, args.month, sample_limit=args.sample_limit)
