"""
Lightweight, canonical artifact loader for serving.

Phase 2: Full loader implementation.
This replaces the initial skeleton with a concrete, cached loader that reads
artifacts from artifacts/run-<timestamp>/ and honors a config-based pointer.
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


_cache = {
    "nn_model": None,
    "embeddings": None,
    "lookup_df": None,
    "feature_columns": None,
}


def _load_artifact(name: str):
    path = _current_run_path() / name
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    return joblib_load(path)


def _load_all_cached():
    global _cache
    if _cache["nn_model"] is None:
        _cache["nn_model"] = _load_artifact("nn_model.pkl")
    if _cache["embeddings"] is None:
        _cache["embeddings"] = _load_artifact("embeddings.pkl")
    if _cache["lookup_df"] is None:
        _cache["lookup_df"] = _load_artifact("df_lookup.pkl")
    if _cache["feature_columns"] is None:
        # feature_columns.json is read directly (not a joblib dump)
        feat_path = _current_run_path() / "feature_columns.json"
        if feat_path.exists():
            with open(feat_path, "r") as f:
                _cache["feature_columns"] = _json_load(f)
        else:
            _cache["feature_columns"] = None
    return _cache["nn_model"], _cache["embeddings"], _cache["lookup_df"], _cache["feature_columns"]


def _json_load(fp):
    import json
    return json.load(fp)


def get_nn_model():
    nn, _, _, _ = _load_all_cached()
    return nn


def get_embeddings():
    _, emb, _, _ = _load_all_cached()
    return emb


def get_lookup_df():
    _, _, df, _ = _load_all_cached()
    return df


def get_feature_columns():
    _, _, _, cols = _load_all_cached()
    return cols


__all__ = ["get_nn_model", "get_embeddings", "get_lookup_df", "get_feature_columns"]
