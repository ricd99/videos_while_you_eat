from __future__ import annotations

import json
from huggingface_hub import hf_hub_download
from joblib import load as joblib_load
from src.core.config import settings


def _download_artifact(filename: str) -> str:
    """Download a single artifact from Hugging Face Hub with caching."""
    return hf_hub_download(
        repo_id=settings.hf_repo_id,
        filename=filename,
    )


def _load_artifact(name: str):
    """Load artifact from Hugging Face Hub (with local cache)."""
    path = _download_artifact(name)
    return joblib_load(path)


def _load_all_cached():
    """Load all artifacts, downloading from HF Hub if needed."""
    global _cache

    if _cache["nn_model"] is None:
        _cache["nn_model"] = _load_artifact("nn_model.pkl")

    if _cache["embeddings"] is None:
        _cache["embeddings"] = _load_artifact("embeddings.pkl")

    if _cache["lookup_df"] is None:
        _cache["lookup_df"] = _load_artifact("df_lookup.pkl")

    if _cache["feature_columns"] is None:
        path = _download_artifact("feature_columns.json")
        with open(path, "r") as f:
            _cache["feature_columns"] = json.load(f)

    return _cache["nn_model"], _cache["embeddings"], _cache["lookup_df"], _cache["feature_columns"]


_cache = {
    "nn_model": None,
    "embeddings": None,
    "lookup_df": None,
    "feature_columns": None,
}


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
