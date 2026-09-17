#!/usr/bin/env python3
"""
Runs sequentially: collect → ETL → train → evaluate
"""
import argparse
import json
import joblib
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from huggingface_hub import HfApi
from sklearn.model_selection import train_test_split

# Allows imports from src/ directory structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.data_pipeline.collect_channels import collect
from scripts.data_pipeline.etl import run_etl
from src.core.config import settings
from src.core.failure_tracker import FailureTracker
from src.core.pipeline_state import PipelineState
from src.db.connection import db_manager
from src.embedding import batch_encode
from src.features.build_features import build_features
from src.models.evaluate import evaluate_model
from src.models.train import train_model
from src.models.tune import tune_model
from src.utils.validate_data import validate_data


def main(args):
    """Main training pipeline function that orchestrates the complete ML workflow."""
    project_root = Path(__file__).resolve().parent.parent
    pipeline_start_time = time.time()

    pipeline_state = PipelineState()
    failure_tracker = FailureTracker()
    run_id = datetime.now().strftime("run-%Y%m%d-%H%M%S")
    pipeline_state.start_run(run_id)

    try:
        print("collecting new channels from yt api")
        stage_start = time.time()
        collect()
        pipeline_state.complete_stage("collect")

        print("running etl")
        stage_start = time.time()
        new_channel_count = run_etl()
        pipeline_state.set_new_channel_count(new_channel_count)
        pipeline_state.complete_stage("etl", metadata={"new_channel_count": new_channel_count})

        if new_channel_count == 0:
            print("no new channels found. skipping model training")
            pipeline_state.reset()
            return

        print("Loading data from RDS...")
        stage_start = time.time()
        df_enc = db_manager.fetch_dataframe("SELECT * FROM channels_final")
        print(f"Data loaded: {df_enc.shape[0]} rows, {df_enc.shape[1]} columns")
        pipeline_state.complete_stage("load")

        # Save feature metadata for serving consistency
        artifacts_dir = os.path.join(project_root, "artifacts", run_id)
        os.makedirs(artifacts_dir, exist_ok=True)

        # Save run_id for local development
        latest_run_path = os.path.join(project_root, "artifacts", "latest_run.txt")
        with open(latest_run_path, "w") as f:
            f.write(run_id)

        feature_cols = list(df_enc.columns)
        with open(os.path.join(artifacts_dir, "feature_columns.json"), "w") as f:
            json.dump(feature_cols, f)
        print(f"Saved {len(feature_cols)} feature columns for serving consistency")

        print("Validating data quality...")
        stage_start = time.time()
        is_valid, failed = validate_data(df_enc)
        pipeline_state.record_validation(is_valid, failed)
        pipeline_state.complete_stage("validate")

        # Data drift detection
        failure_rate = pipeline_state.get_validation_failure_rate(last_n=10)
        if failure_rate > 0.2:
            print(f"WARNING: Data drift detected! Validation failure rate: {failure_rate:.1%}")

        if not is_valid:
            pipeline_state.reset()
            raise ValueError(f"Data quality check failed. Issues: {failed}")
        print("Data validation passed.")

        print("Training Nearest Neighbors model...")
        df_train, df_test = train_test_split(df_enc, train_size=0.98, random_state=67)
        df_train = df_train.reset_index(drop=True)
        df_test = df_test.reset_index(drop=True)
        print(f"Train: {df_train.shape[0]} samples | Test: {df_test.shape[0]} samples")

        # Hyperparameters come from settings; tune_model is a thin wrapper kept
        # for symmetry with the rest of the pipeline.
        stage_start = time.time()
        best_params = tune_model(df_train, df_test)
        print(f"Hyperparameters: {best_params}")
        pipeline_state.complete_stage("train", metadata={"best_params": best_params})

        print("Training optimized Nearest Neighbors model...")
        stage_start = time.time()
        nn, embeddings, df_lookup = train_model(df_train, params=best_params)
        train_time = time.time() - stage_start
        print(f"Model trained in {train_time:.2f} seconds")

        print("Evaluating model...")
        stage_start = time.time()
        test_texts = df_test["text"].fillna("").tolist()
        test_embeddings = batch_encode(test_texts)
        mean_dist, median_dist = evaluate_model(nn, test_embeddings)
        print(f"Mean distance: {mean_dist}")
        print(f"Median distance: {median_dist}")
        pipeline_state.complete_stage("evaluate", metadata={"mean_dist": mean_dist, "median_dist": median_dist})

        print("Saving model...")
        stage_start = time.time()
        joblib.dump(nn, os.path.join(artifacts_dir, "nn_model.pkl"))
        joblib.dump(embeddings, os.path.join(artifacts_dir, "embeddings.pkl"))
        joblib.dump(df_lookup, os.path.join(artifacts_dir, "df_lookup.pkl"))
        print(f"Model and embeddings and lookup table saved to {artifacts_dir}")
        pipeline_state.complete_stage("save", metadata={"artifacts_dir": artifacts_dir})

        print("Uploading artifacts to Hugging Face Hub...")
        stage_start = time.time()

        api = HfApi()

        # Upload each artifact
        artifacts_to_upload = ["nn_model.pkl", "embeddings.pkl", "df_lookup.pkl", "feature_columns.json"]
        for filename in artifacts_to_upload:
            filepath = os.path.join(artifacts_dir, filename)
            if os.path.exists(filepath):
                api.upload_file(
                    path_or_fileobj=filepath,
                    path_in_repo=filename,
                    repo_id=settings.hf_repo_id,
                )
                print(f"Uploaded {filename} to {settings.hf_repo_id}")

        print("Artifacts uploaded to Hugging Face Hub")
        pipeline_state.complete_stage("upload")

        total_time = time.time() - pipeline_start_time
        print(f"Pipeline completed in {total_time:.2f} seconds")
        pipeline_state.reset()

    except Exception as e:
        stage = pipeline_state.state.get("last_successful_stage", "collect")
        error_msg = str(e)
        print(f"Pipeline failed at stage: {stage}")
        print(f"Error: {error_msg}")

        # Log failure to S3 (not GitHub Issues)
        failure_tracker.log_failure(
            stage=stage,
            error=error_msg,
            run_id=run_id,
            metadata={"new_channel_count": pipeline_state.state.get("new_channel_count", 0)}
        )

        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run youtube pipeline with NearestNeighbors + Sentence Embeddings + MLflow")
    p.add_argument("--experiment", type=str, default="Youtube Recommender")

    args = p.parse_args()
    main(args)