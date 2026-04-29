#!/usr/bin/env python3
"""
Runs sequentially: load → validate → preprocess → feature engineering
"""

import os
import sys
import time
import argparse
import mlflow
from sklearn.model_selection import train_test_split

# Allows imports from src/ directory structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.data_pipeline.collect_channels import collect
from scripts.data_pipeline.etl import run_etl
from src.db.connection import db_manager

from src.data.preprocess_data import preprocess_data
from src.features.build_features import build_features
from src.utils.validate_data import validate_data   
from src.embedding import batch_encode
from src.models.train import train_model
from src.models.tune import tune_model
from src.models.evaluate import evaluate_model
from pathlib import Path



def main(args):
    """
    Main training pipeline function that orchestrates the complete ML workflow.
    
    """
    project_root = Path(__file__).resolve().parent.parent
    mlruns_path = project_root / "mlruns"

    mlflow.set_tracking_uri(mlruns_path.as_uri())
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run():
        # === STAGE 0: Collect New Channels ===
        print("collecting new channels from yt api")
        collect()

         # === STAGE 0.5: ETL  ===
        print("running etl")
        new_channel_count = run_etl()

        if new_channel_count == 0:
            print("no new channels found. skipping model training")
            return

        # === STAGE 1: Data Loading ===
        print("Loading data from RDS...")
        df = db_manager.fetch_dataframe("SELECT * FROM channels_final")
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")


        # === STAGE 2: Data Preprocessing ===
        print("Preprocessing data...")
        df_pp = preprocess_data(df) 
    
        processed_path = os.path.join(project_root, "data", "processed", "channels_pp.csv") # Save processed dataset for reproducibility and debugging
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        df_pp.to_csv(processed_path, index=False)
        print(f"Processed dataset saved to {processed_path} | Shape: {df_pp.shape}")

        # === STAGE 3: Feature Engineering ===
        print("Building features...")
        df_enc = build_features(df_pp) 
        print(f"Feature engineering completed: {df_enc.shape[1]} features")

        #Save Feature Metadata for Serving Consistency
        # This ensures serving pipeline uses exact same features in exact same order
        import json, joblib
        artifacts_dir = os.path.join(project_root, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)

        feature_cols = list(df_enc.columns)
        
        with open(os.path.join(artifacts_dir, "feature_columns.json"), "w") as f:
            json.dump(feature_cols, f)

        mlflow.log_text("\n".join(feature_cols), artifact_file="feature_columns.txt")

        preprocessing_artifact = {
            "feature_columns": feature_cols,  
        }
        joblib.dump(preprocessing_artifact, os.path.join(artifacts_dir, "preprocessing.pkl"))
        mlflow.log_artifact(os.path.join(artifacts_dir, "preprocessing.pkl"))
        print(f"Saved {len(feature_cols)} feature columns for serving consistency")


        # === STAGE 3.1: Data Validation (right before trainng model) ===
        print("Validating data quality with Great Expectations...")
        is_valid, failed = validate_data(df_enc)
        mlflow.log_metric("data_quality_pass", int(is_valid))

        if not is_valid:
            import json
            mlflow.log_text(json.dumps(failed, indent=2), artifact_file="failed_expectations.json")
            raise ValueError(f"Data quality check failed. Issues: {failed}")
        else:
            print("Data validation passed. Logged to MLflow.")


        # === STAGE 4: Model Training and Tuning ===
        print("Training and tuning NearestNeighbors model...")

        df_train, df_test = train_test_split(df_enc, train_size=0.98, random_state=67)
        df_train = df_train.reset_index(drop=True) 
        df_test = df_test.reset_index(drop=True)
        print(f"Train: {df_train.shape[0]} samples | Test: {df_test.shape[0]} samples")

        best_params = tune_model(df_train, df_test)
        mlflow.log_params(best_params)
        print(f"Tuning complete. Best params: {best_params}")


        # === STAGE 5: Final Training ===
        print("Training optimized Nearest Neighbors model...")
        t0 = time.time()
        nn, embeddings, df_lookup = train_model(df_train, params=best_params)
        train_time = time.time() - t0
        mlflow.log_metric("train_time", train_time)
        print(f"Model trained in {train_time:.2f} seconds")

        
        # === STAGE 6: Evaluating===
        print("Evaluating model...")
        test_texts = df_test["text"].fillna("").tolist()
        test_embeddings = batch_encode(test_texts)
        mean_dist, median_dist = evaluate_model(nn, test_embeddings)
        mlflow.log_metric("mean_nn_distance", mean_dist)
        mlflow.log_metric("median_nn_distance", median_dist)
        print(f"Mean distance: {mean_dist}")
        print(f"Median distnace: {median_dist}")


        # === STAGE 7: Save Model===
        print("Saving model...")
        artifacts_dir = os.path.join(project_root, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)

        joblib.dump(nn, os.path.join(artifacts_dir, "nn_model.pkl"))
        joblib.dump(embeddings, os.path.join(artifacts_dir, "embeddings.pkl"))
        joblib.dump(df_lookup, os.path.join(artifacts_dir, "df_lookup.pkl"))


        mlflow.log_artifact(os.path.join(artifacts_dir, "nn_model.pkl"))
        mlflow.log_artifact(os.path.join(artifacts_dir, "embeddings.pkl"))
        mlflow.log_artifact(os.path.join(artifacts_dir, "df_lookup.pkl"))
        print("Model and emebddings, and lookup table saved")

       


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run youtube pipeline with NewarestNeighbors + MLflow")
    p.add_argument("--experiment", type=str, default="Youtube Recommender")
    p.add_argument("--mlflow_uri", type=str, default=None)

    args = p.parse_args()
    main(args)

"""
# Use this below to run the pipeline:

python scripts/run_pipeline.py

"""