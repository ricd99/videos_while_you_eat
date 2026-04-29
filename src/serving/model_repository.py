"""
Lightweight, canonical artifact loader for serving.

Phase 1: Skeleton implementation. The real loading logic will be filled in Phase 2.
"""
from __future__ import annotations

import json
from pathlib import Path

from joblib import load as joblib_load
from src.config import settings


def _current_run_id() -> str:
    # Prefer explicit pointer from config
    if getattr(settings, "latest_artifacts_run", None):
        return settings.latest_artifacts_run

    # Fallback to a lightweight latest_run.txt in the artifacts root
    artifacts_root = Path(getattr(settings, "artifacts_root", Path.cwd() / "artifacts"))
    latest_path = artifacts_root / "latest_run.txt"
    if latest_path.exists():
        try:
            return latest_path.read_text().strip()
        except Exception:
            pass
    # Final fallback
    return "run-default"


def _current_run_path() -> Path:
    root = Path(getattr(settings, "artifacts_root", Path.cwd() / "artifacts"))
    run_id = _current_run_id()
    return root / run_id


def _load_artifact(name: str):
    path = _current_run_path() / name
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    return joblib_load(path)


def get_nn_model():
    return _load_artifact("nn_model.pkl")


def get_embeddings():
    return _load_artifact("embeddings.pkl")


def get_lookup_df():
    return _load_artifact("df_lookup.pkl")


def get_feature_columns():
    p = _current_run_path() / "feature_columns.json"
    if p.exists():
        with open(p, "r") as f:
            return json.load(f)
    return None


__all__ = ["get_nn_model", "get_embeddings", "get_lookup_df", "get_feature_columns"]
