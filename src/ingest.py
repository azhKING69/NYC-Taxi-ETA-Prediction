import argparse
import sys
import urllib.request
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import BRONZE_DIR, TLC_BASE_URL


def download_monthly_data(year: int, month: int, dest_dir: Path = BRONZE_DIR) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = f"yellow_tripdata_{year}-{month:02d}.parquet"
    target_path = dest_dir / filename

    if target_path.exists() and target_path.stat().st_size > 0:
        print(f"Data file already exists: {target_path} ({target_path.stat().st_size / (1024 * 1024):.2f} MB)")
        return target_path

    url = f"{TLC_BASE_URL}/{filename}"
    print(f"Downloading from {url} to {target_path}...")
    try:
        urllib.request.urlretrieve(url, target_path)
        print(f"Download complete: {target_path} ({target_path.stat().st_size / (1024 * 1024):.2f} MB)")
        return target_path
    except Exception as e:
        if target_path.exists():
            target_path.unlink()
        raise RuntimeError(f"Failed to download {url}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest NYC TLC Yellow Taxi data")
    parser.add_argument("--year", type=int, default=2026, help="Data year (e.g. 2026)")
    parser.add_argument("--month", type=int, default=5, help="Data month (e.g. 5 for May)")
    args = parser.parse_args()

    download_monthly_data(args.year, args.month)
