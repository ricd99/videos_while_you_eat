import json
import pytest
import joblib
from pathlib import Path

import pandas as pd
import numpy as np

from src.core.config import settings


def _prepare_run(tmp_path: Path, run_id: str):
    run_dir = tmp_path / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    # nn_model.pkl
    nn_sample = {"name": "dummy_nn"}
    joblib.dump(nn_sample, run_dir / "nn_model.pkl")
    # embeddings.pkl
    joblib.dump(np.array([[1.0, 2.0, 3.0]]), run_dir / "embeddings.pkl")
    # df_lookup.pkl
    df_lookup = pd.DataFrame([{"channel_name": "A", "channel_id": "idA"}])
    joblib.dump(df_lookup, run_dir / "df_lookup.pkl")
    # feature_columns.json
    with open(run_dir / "feature_columns.json", "w") as f:
        json.dump(["channel_id", "channel_name", "text"], f)
    return run_dir


@pytest.mark.parametrize("multiple_run", [False])
def test_model_repository_loads_from_run_folder(tmp_path, monkeypatch, multiple_run):
    # Prepare a canonical per-run artifacts folder in a temp directory
    run_id = "run-test"
    run_dir = _prepare_run(tmp_path, run_id)

    # Point config to use the temp artifacts root and the specific run
    settings.artifacts_root = tmp_path
    settings.latest_artifacts_run = run_id

    # Import after setting config to ensure loader resolves path
    from src.serving.model_repository import (
        get_nn_model,
        get_lookup_df,
        get_embeddings,
        get_feature_columns,
    )

    nn = get_nn_model()
    df_lookup = get_lookup_df()
    emb = get_embeddings()
    cols = get_feature_columns()

    assert isinstance(nn, dict)
    assert isinstance(df_lookup, pd.DataFrame)
    assert isinstance(emb, np.ndarray) or hasattr(emb, "shape")
    assert isinstance(cols, list)
