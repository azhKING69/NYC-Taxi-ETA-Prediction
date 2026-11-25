import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import pytest
import pandera as pa

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.validate import get_bronze_schema


def test_bronze_schema_valid_data():
    schema = get_bronze_schema()
    valid_df = pd.DataFrame({
        "tpep_pickup_datetime": [pd.Timestamp("2026-05-15 08:30:00")],
        "tpep_dropoff_datetime": [pd.Timestamp("2026-05-15 08:55:00")],
        "PULocationID": [161],
        "DOLocationID": [237],
        "trip_distance": [3.2],
        "passenger_count": [1.0],
        "fare_amount": [18.5],
    })
    validated = schema.validate(valid_df)
    assert len(validated) == 1


def test_bronze_schema_catches_invalid_dates():
    schema = get_bronze_schema()
    invalid_date_df = pd.DataFrame({
        "tpep_pickup_datetime": [pd.Timestamp("2008-12-31 23:59:00")],  # Faulty taxi meter year
        "tpep_dropoff_datetime": [pd.Timestamp("2009-01-01 00:20:00")],
        "PULocationID": [161],
        "DOLocationID": [237],
        "trip_distance": [3.2],
        "passenger_count": [1.0],
        "fare_amount": [18.5],
    })
    with pytest.raises(pa.errors.SchemaError):
        schema.validate(invalid_date_df)


def test_bronze_schema_catches_invalid_zone_ids():
    schema = get_bronze_schema()
    invalid_zone_df = pd.DataFrame({
        "tpep_pickup_datetime": [pd.Timestamp("2026-05-15 08:30:00")],
        "tpep_dropoff_datetime": [pd.Timestamp("2026-05-15 08:55:00")],
        "PULocationID": [999],  # Valid TLC zones are 1 to 265
        "DOLocationID": [237],
        "trip_distance": [3.2],
        "passenger_count": [1.0],
        "fare_amount": [18.5],
    })
    with pytest.raises(pa.errors.SchemaError):
        schema.validate(invalid_zone_df)
