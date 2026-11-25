from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
MODEL_ARTIFACT_DIR = BASE_DIR / "model_artifact"

# NYC TLC Data URL
TLC_BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"

# MLflow
MLFLOW_TRACKING_URI = f"sqlite:///{BASE_DIR / 'mlflow.db'}"
MLFLOW_EXPERIMENT_NAME = "nyc-taxi-trip-duration"
REGISTERED_MODEL_NAME = "nyc-taxi-duration-predictor"

# Features & Target
TARGET_COL = "trip_duration_minutes"
CATEGORICAL_FEATURES = ["PULocationID", "DOLocationID"]
NUMERICAL_FEATURES = [
    "trip_distance",
    "passenger_count",
    "pickup_hour",
    "pickup_day_of_week",
    "is_weekend",
    "is_rush_hour",
    "is_airport_trip",
]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

# Validation & Cleaning Boundaries
MIN_TRIP_DISTANCE = 0.1
MAX_TRIP_DISTANCE = 100.0
MIN_TRIP_DURATION = 1.0     # minutes
MAX_TRIP_DURATION = 120.0   # minutes
MIN_FARE_AMOUNT = 2.50
MAX_FARE_AMOUNT = 1000.0
VALID_ZONE_MIN = 1
VALID_ZONE_MAX = 265
AIRPORT_ZONES = [1, 132, 138]  # 1: Newark, 132: JFK, 138: LaGuardia
