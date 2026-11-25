import sys
from pathlib import Path
from datetime import datetime
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))
from serving.app import TripPredictionRequest, transform_request_to_features


def test_rush_hour_extraction():
    # Weekday 8:30 AM is rush hour
    req_rush = TripPredictionRequest(
        PULocationID=161,
        DOLocationID=237,
        trip_distance=3.0,
        pickup_datetime=datetime(2026, 5, 15, 8, 30),  # Friday
        passenger_count=1,
    )
    features_rush = transform_request_to_features(req_rush)
    assert features_rush["is_rush_hour"] == 1
    assert features_rush["is_weekend"] == 0

    # Sunday 8:30 AM is weekend, NOT rush hour
    req_weekend = TripPredictionRequest(
        PULocationID=161,
        DOLocationID=237,
        trip_distance=3.0,
        pickup_datetime=datetime(2026, 5, 17, 8, 30),  # Sunday
        passenger_count=1,
    )
    features_weekend = transform_request_to_features(req_weekend)
    assert features_weekend["is_rush_hour"] == 0
    assert features_weekend["is_weekend"] == 1


def test_airport_trip_flag():
    # JFK is zone 132
    req_airport = TripPredictionRequest(
        PULocationID=161,
        DOLocationID=132,
        trip_distance=16.5,
        pickup_datetime=datetime(2026, 5, 15, 12, 0),
        passenger_count=2,
    )
    features = transform_request_to_features(req_airport)
    assert features["is_airport_trip"] == 1

    # Standard non-airport trip
    req_regular = TripPredictionRequest(
        PULocationID=161,
        DOLocationID=237,
        trip_distance=2.1,
        pickup_datetime=datetime(2026, 5, 15, 12, 0),
        passenger_count=1,
    )
    features_regular = transform_request_to_features(req_regular)
    assert features_regular["is_airport_trip"] == 0
