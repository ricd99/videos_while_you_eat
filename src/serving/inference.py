import joblib
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import faiss

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent    
load_dotenv()

from src.db.connection import db_manager
from src.data.fetch_data_given_query_channel import _get_channel_data
from src.data.preprocess_data import preprocess_data
from src.features.build_features import build_features
from src.embedding import batch_encode
from serving.artifact_loader import get_nn_model, get_lookup_df, get_feature_columns

# Lazy loading - loaded on first request, not at import
_nn = None
_lookup_df = None

def _load_artifacts():
    """Load model artifacts on first request using the canonical loader."""
    global _nn, _lookup_df
    if _nn is None:
        _nn = get_nn_model()
    if _lookup_df is None:
        _lookup_df = get_lookup_df()
    return _nn, _lookup_df



def predict(channel_name: str) -> list[dict] | dict:
    nn, df_lookup = _load_artifacts()

    input_dict = _get_channel_data(channel_name)
    if input_dict is None:
        return {"error": f"could not find channel: {channel_name}"}

    df = pd.DataFrame([input_dict])
    df = preprocess_data(df)
    df = build_features(df)
    
    texts = df["text"].fillna("").tolist()
    embedding = batch_encode(texts)

    distances, indices = nn.kneighbors(embedding)

    results = df_lookup.iloc[indices[0]][["channel_name", "channel_id"]].copy()
    results["similarity_score"] = distances[0]
    return results.to_dict(orient="records")
