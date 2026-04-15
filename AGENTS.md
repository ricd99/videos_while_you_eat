# ytrec — YouTube Channel Recommender

## What this is

A nearest-neighbors model trained on long-form video-essay YouTube channels. Given a channel name, it fetches live data from the YouTube API and returns the 5 most similar channels.

## Stack

- FastAPI + Gradio (serving)
- scikit-learn `NearestNeighbors` (model)
- CountVectorizer for text features
- PostgreSQL (RDS) for channel lookup at inference time
- MLflow for experiment tracking (local `mlruns/`)
- Docker: `docker compose up --build`

## Running the app

```bash
# Dev
python -m uvicorn src.app.main:app --reload --port 8000

# Docker (serves on port 8000)
docker compose up --build
```

## Running the ML pipeline

```bash
# Full pipeline: load → preprocess → features → tune → train → evaluate
python scripts/run_pipeline.py --input data/processed/ve_channels/ve_with_features.json

# Test data pipeline only (no training)
python scripts/test_pipelines_phase1_data_features.py

# Prepare processed data from raw JSON
python scripts/prepare_processed_data.py
```

## Critical quirks

- `src/serving/inference.py` uses `Path.cwd()` for model artifact paths — **must run from project root**, not from `src/serving/`.
- `scripts/run_pipeline.py` adds `src/` to `sys.path`; other scripts do the same. Don't assume `src` is on the path.
- Model artifacts are stored in two places:
  - `artifacts/nn_model.pkl`, `artifacts/column_transformer.pkl` (checked in)
  - `src/serving/model/<run_id>/artifacts/` (MLflow copies here)
- Feature pipeline order is **load → preprocess_data → build_features**. `inference.py` uses these same functions.
- The `column_transformer` drops `channel_id` and `channel_name` columns — the `text` column is the only thing vectorized.

## Environment variables

See `.env` (API keys, RDS credentials, AWS keys). Required for inference:
- `YOUTUBE_API_KEY_*` (one of several keys, `ELI` is used in `fetch_data_given_query_channel.py`)
- `RDS_HOST`, `RDS_PW`

## Deployment

CI pushes to Docker Hub on every push to `main` (`ricd99/ytrec:latest`). Container runs uvicorn on port 8000.

## Project structure

```
src/app/main.py         # FastAPI + Gradio entrypoint
src/serving/inference.py # predict() function (model + DB lookup)
src/data/               # load_data, preprocess_data, fetch_data_given_query_channel
src/features/           # build_features (text concatenation)
src/models/             # train, tune (Optuna), evaluate
scripts/                # pipeline runners, database setup
artifacts/              # nn_model.pkl, column_transformer.pkl, feature_columns.json
data/processed/ve_channels/ve_with_features.json  # training data
mlruns/                  # MLflow tracking
```
