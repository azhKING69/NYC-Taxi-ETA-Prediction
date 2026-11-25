# NYC Taxi Trip Duration Predictor — Full MLOps Pipeline

## The Project in One Sentence

Predict **how long a NYC yellow taxi trip will take** (in minutes) given the pickup location, dropoff location, time of day, and day of week — with a fully automated pipeline that retrains monthly on fresh government data.

---

## Your Dataset (Already Downloaded)

You have `yellow_tripdata_2026-05.parquet` sitting in your MLops folder. Here's what's inside:

| Fact | Value |
|---|---|
| **Rows** | **4,090,836** (4 million trips in a single month!) |
| **Columns** | 20 |
| **File size** | 69.7 MB |
| **Source** | NYC Taxi & Limousine Commission (TLC) |
| **Download URL pattern** | `https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_YYYY-MM.parquet` |
| **Update frequency** | **New file every month**, automatically, forever |
| **Auth required** | None. Zero. Just `wget` the URL. |

### Key columns you'll use

| Column | What it means | Role |
|---|---|---|
| `tpep_pickup_datetime` | When the passenger got in | Feature (extract hour, day_of_week, month) |
| `tpep_dropoff_datetime` | When the passenger got out | Used to compute the **target** |
| `PULocationID` | Pickup zone (1–265, maps to neighborhoods) | Feature |
| `DOLocationID` | Dropoff zone (1–265) | Feature |
| `trip_distance` | Miles driven (odometer) | Feature |
| `passenger_count` | Number of passengers | Feature |
| `fare_amount` | The fare charged | Can be a feature or ignored |

### The target variable (what you predict)

```
trip_duration_minutes = (tpep_dropoff_datetime - tpep_pickup_datetime).total_seconds() / 60
```

This is **not** a column in the raw data — you **engineer** it from the two datetime columns. This is a great thing to explain in an interview: "The raw data doesn't hand you the label; you derive it."

---

## Is This Worth a Resume Project? Absolutely Yes.

Here's why this is one of the strongest possible MLOps portfolio projects:

1. **Real, government-published data** — not a Kaggle toy dataset. NYC TLC is used by Uber, Lyft, and academic researchers.
2. **Genuinely needs retraining** — traffic patterns change by season, holidays, construction, and policy (e.g., the new CBD congestion fee column that appeared in your data!). A model trained on January data *will* degrade by July. This is real drift, not simulated.
3. **4 million rows per month** — big enough to be taken seriously, small enough to train on a laptop.
4. **The "new data" story is airtight** — a new Parquet file appears at a predictable URL every month. Your cron job just downloads it. No APIs, no scraping, no authentication.
5. **Industry-relevant** — Uber, Lyft, DoorDash, and every ride-hailing company runs this exact prediction. Interviewers will immediately understand the problem.

---

## The Full Pipeline, Stage by Stage

Here's how each stage from the MLOps diagram maps to concrete decisions for this project:

### [1] Data Ingestion → Raw Data Store

**What happens:** A script downloads the latest monthly Parquet file from the NYC TLC website and saves it into the `data/bronze/` folder, untouched.

**Where data is stored:**
- **For this project:** Local disk, inside `data/bronze/`. Each monthly file is saved with its original name (`yellow_tripdata_2026-05.parquet`).
- **In production:** You'd swap this for an S3 bucket. The script would do `aws s3 cp` instead of saving locally. But for a resume project, local disk is completely fine.

**Key design choice:** Never modify the bronze file. It's the raw, untouched truth. If your cleaning logic has a bug, you can always re-derive everything from bronze.

```python
# src/ingest.py
import urllib.request
from pathlib import Path

def download_month(year: int, month: int, dest_dir: str = "data/bronze"):
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet"
    dest = Path(dest_dir) / f"yellow_tripdata_{year}-{month:02d}.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    return dest
```

---

### [2] Data Validation (Schema Checks, Drift Checks)

**What happens:** Before we touch the data, we verify it hasn't changed format (schema validation) and doesn't contain garbage (range checks).

**Tool:** Pandera (lightweight, Python-native).

**What we validate from your actual data:**
- `tpep_pickup_datetime` and `tpep_dropoff_datetime` are actual datetimes (not null, not in the year 2008 — yes, your data has some!)
- `PULocationID` and `DOLocationID` are integers between 1 and 265
- `trip_distance` ≥ 0 (your data has a max of 307,491 miles — clearly garbage that needs catching)
- `passenger_count` between 0 and 9
- `fare_amount` ≥ 0 (your data has negative fares — another thing to catch)

**Why this matters for interviews:** ~955,000 rows (23% of your data!) have nulls in `passenger_count`, `RatecodeID`, `congestion_surcharge`, and `Airport_fee`. Being able to say "I built validation that catches these before training" shows real engineering maturity.

---

### [3] Data Cleaning + EDA

**What happens:** Fix the problems validation identified. This produces the `data/silver/` layer.

**Cleaning steps specific to your data:**
1. **Filter out insane dates** — trips from 2008/2009 appearing in a 2026 file (your data has `min pickup = 2008-12-31`)
2. **Filter out negative fares** — fare_amount min is -950 in your data
3. **Filter out impossible distances** — max trip_distance is 307,491 miles (that's to the moon)
4. **Filter out impossible durations** — trips lasting 0 seconds or 24+ hours
5. **Fill nulls** — `passenger_count` nulls → fill with 1 (most common value)
6. **Compute the target** — `trip_duration_minutes = (dropoff - pickup).total_seconds() / 60`
7. **Filter duration outliers** — keep trips between 1 minute and 120 minutes (reasonable taxi rides)

**EDA (done once in a notebook, not in the pipeline):**
- Distribution of trip durations (you'll see it's right-skewed, peaking around 10-15 minutes)
- Heatmap of pickup/dropoff zones (Manhattan dominates)
- Duration by hour of day (rush hour is slower)
- Duration by day of week (weekends vs weekdays)

---

### [4] Feature Engineering → Processed Data Store

**What happens:** Transform raw columns into features the model can learn from. Output goes to `data/gold/`.

**Features you'll engineer:**

| Feature | How | Why |
|---|---|---|
| `hour` | Extract from pickup datetime | Rush hour trips take longer |
| `day_of_week` | Extract (0=Mon, 6=Sun) | Weekend vs weekday patterns |
| `is_weekend` | 1 if Sat/Sun | Binary version of above |
| `month` | Extract | Seasonal patterns |
| `PULocationID` | As-is (categorical) | Pickup neighborhood matters hugely |
| `DOLocationID` | As-is (categorical) | Dropoff neighborhood matters hugely |
| `trip_distance` | As-is | Longer distance = longer trip (usually) |
| `passenger_count` | As-is | Minor effect but real |
| `is_airport` | 1 if PU/DO is JFK (132) or LaGuardia (138) | Airport trips have distinct patterns |
| `is_rush_hour` | 1 if hour in [7-10, 16-19] on weekdays | Traffic congestion signal |

**Data versioning (DVC):**
- For this project: Git + DVC, with the actual Parquet files stored locally or on Google Drive (free).
- Each month's processed data gets a new DVC pointer commit, so you can always trace "model v3 was trained on data from months Jan-May 2026."

**Train/Test Split:**
- **Training set:** All trips except a random 10% held-out sample
- **Frozen test set:** That 10% holdout, saved once and never touched again. Versioned with DVC.
- **Important:** Split is done *randomly within a month*, but the monthly retraining adds *new months* to the training set. The frozen test set from month 1 stays frozen forever.

---

### [5] Training Pipeline → MLflow Experiment Tracking

**What model?**

**LightGBM** (gradient-boosted decision trees). Here's why, not some other model:

| Model | Why yes / why no |
|---|---|
| Linear Regression | ❌ Too simple. Can't capture nonlinear interactions (rush hour + airport + distance). |
| Random Forest | ⚠️ Works, but slow to train on 4M rows and doesn't handle categoricals natively. |
| **LightGBM** | ✅ **Best choice.** Fast on 4M rows, handles categoricals natively (LocationIDs), excellent accuracy on tabular data, industry standard for this exact type of problem. |
| XGBoost | ✅ Also good, but LightGBM is faster on large data and handles categoricals better. |
| Neural Network (PyTorch) | ❌ Overkill for tabular data. Harder to explain, slower to train, usually worse accuracy than gradient boosting on structured data. |
| Deep Learning (Transformers) | ❌ Absurd for this problem. Shows you don't understand when to use what. |

**Training flow:**
1. Load gold-layer train data
2. Sweep 3-5 hyperparameter configs (learning rate, max depth, num leaves, num trees)
3. Each config is an MLflow run: log params, metrics (RMSE, MAE, R²), and the trained model artifact
4. Pick the best run by validation RMSE

**MLflow stores:**
- **Tracking database:** SQLite locally (`mlflow.db`), Postgres in production
- **Artifact store:** Local `models/mlruns/` folder, S3 in production
- Every run logs: `data_version` (which months), all hyperparams, val_rmse, val_mae, val_r2, the model file

```python
# Pseudocode for training
import lightgbm as lgb
import mlflow

with mlflow.start_run():
    mlflow.log_param("data_months", "2026-01 to 2026-05")
    mlflow.log_params({"learning_rate": 0.1, "num_leaves": 63, "n_estimators": 500})
    
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train, categorical_feature=["PULocationID", "DOLocationID"])
    
    preds = model.predict(X_val)
    rmse = mean_squared_error(y_val, preds, squared=False)
    mlflow.log_metric("val_rmse", rmse)
    mlflow.lightgbm.log_model(model, "model")  # native MLflow integration!
```

---

### [6] Evaluation Pipeline → Frozen Test Set

**What happens:** The best model from the sweep is tested against the frozen test set. This is the **gate**.

**Metrics:**
- **RMSE** (Root Mean Squared Error) — primary metric. "On average, the model's prediction is off by X minutes."
- **MAE** (Mean Absolute Error) — secondary. More interpretable: "The median error is Y minutes."
- **R²** — how much variance in duration the model explains

**Gate logic:**
```
IF test_rmse < 5.0 minutes → PASS → promote to Staging
IF test_rmse >= 5.0 → FAIL → pipeline stops, model is NOT registered
```

A 5-minute RMSE on NYC taxi trips is a reasonable and achievable threshold.

---

### [7] Model Registry (Staging → Production)

**What happens:** If evaluation passes, the model is registered in MLflow's Model Registry.

**Lifecycle:**
```
None → Staging (automatic, after evaluation gate passes)
Staging → Production (manual approval, or automatic if you're confident)
Production → Archived (when a newer model replaces it)
```

**What gets stored:** The model artifact, plus metadata: which data version trained it, what its test RMSE was, when it was registered, who approved it.

---

### [8] Deployment (Docker → Server)

**Two Docker images:**

**Training image (`Dockerfile.train`):**
- Contains: Python, LightGBM, MLflow, Pandera, pandas, the `src/` pipeline code
- Runs: As a **job** (starts, trains, exits)
- Trigger: GitHub Actions cron (monthly) or manual

**Serving image (`Dockerfile.serve`):**
- Contains: Python, LightGBM, MLflow (for loading), FastAPI, uvicorn
- Runs: As an **always-on service**
- Exposes: `POST /predict` — send pickup/dropoff/hour/distance, get back predicted duration in minutes

```python
# serving/app.py
@app.post("/predict")
def predict(req: TripPredictionRequest):
    features = engineer_features(req)  # extract hour, day_of_week, is_airport, etc.
    duration = model.predict(features)
    return {"predicted_duration_minutes": round(duration, 1)}
```

**Where to host (free tier options for a resume project):**
- **Render.com** — free tier, deploy a Docker image, get a public URL
- **Railway.app** — similar, slightly more generous
- **Google Cloud Run** — pay-per-request, essentially free at low traffic

---

### [9] Monitoring (Drift, Decay) → Triggers Retraining

**What to monitor:**

| Check | How | Tool |
|---|---|---|
| **Prediction quality** | Compare predicted vs actual duration (if you log actuals) | Custom script + MLflow |
| **Data drift** | Are this month's feature distributions different from training data? (e.g., did a new taxi zone appear?) | Evidently AI or a simple KS-test script |
| **Model staleness** | Has it been > N months since last retrain? | Simple date check in the pipeline |

**The retraining loop:**
```
GitHub Actions cron (1st of every month)
    → Download last month's Parquet from NYC TLC
    → Validate schema
    → Clean + engineer features
    → Retrain on all available months
    → Evaluate against frozen test set
    → If passes → register to Staging
    → (Manual approval) → promote to Production
    → Rebuild & redeploy serving Docker image
```

---

## Complete Architecture Diagram

```
NYC TLC Website (new Parquet file every month)
      │
      ▼
[1] Ingest (download_month.py)  ──▶  data/bronze/yellow_tripdata_YYYY-MM.parquet
      │
      ▼
[2] Validate (Pandera)  ── schema checks, range checks, null checks
      │
      ▼
[3] Clean  ──▶  data/silver/ (filtered, nulls handled, target computed)
      │
      ▼
[4] Feature Engineering  ──▶  data/gold/train.parquet + data/gold/test.parquet (DVC versioned)
      │
      ▼
[5] Train (LightGBM, 3-5 configs)  ──▶  MLflow Tracking (params, metrics, model artifacts)
      │
      ▼
[6] Evaluate (gate: RMSE < 5 min)  ──▶  Frozen test set
      │
      ▼
[7] Registry  ──▶  movie-recommender v1 Staging → Production
      │
      ▼
[8] Deploy  ──▶  Docker (FastAPI) → Render.com / Cloud Run
      │
      ▼
[9] Monitor  ──▶  drift check, staleness check → triggers [1] again next month
```

---

## Tech Stack Summary

| Component | Tool | Why this one |
|---|---|---|
| Language | Python 3.11 | Industry standard for ML |
| Model | LightGBM | Best for tabular regression at this scale |
| Data validation | Pandera | Lightweight, Pythonic, no setup overhead |
| Experiment tracking | MLflow | Industry standard, free, self-hosted |
| Model registry | MLflow Model Registry | Built into MLflow, no extra tool needed |
| Data versioning | DVC + Git | Tracks which exact data trained which model |
| API serving | FastAPI | Fast, auto-docs, type validation |
| Containerization | Docker | Reproducible environments for train + serve |
| CI/CD | GitHub Actions | Free tier, cron scheduling, runs the whole pipeline |
| Hosting | Render.com (free tier) | Zero-config Docker deployment with a public URL |
| Monitoring | Custom script + Evidently (optional) | KS-test for drift, RMSE tracking for decay |

---

## Open Questions

> [!IMPORTANT]
> **Do you want to build this as a completely new project (clean folder), or replace the movie recommender code that's currently in `MLops/`?**

> [!IMPORTANT]  
> **Do you want to start with just the May 2026 file, or should we also download a few more months (e.g., Jan-Apr 2026) so the retraining story is more visible from day one?**

> [!NOTE]
> The movie recommender code (`src/`, `serving/`, `tests/`, etc.) is still in the folder alongside the taxi Parquet file. We can either clean it out or start fresh in a new directory like `~/Desktop/nyc-taxi-mlops/`.
