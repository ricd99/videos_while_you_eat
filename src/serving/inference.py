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
from src.serving.artifact_loader import get_nn_model, get_lookup_df
from src.core.config import settings

# Lazy loading - loaded on first request, not at import
_nn = None
_lookup_df = None

def _load_artifacts():
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
    
    channel_id = input_dict["channel_id"]  # to compare with input channel at end

    df = pd.DataFrame([input_dict])
    df = preprocess_data(df)
    df = build_features(df)

    texts = df["text"].fillna("").tolist()
    embedding = batch_encode(texts)

    # Request TOP_K+1 to account for possible self-match
    top_k = settings.top_k_recommendations
    n_request = min(top_k + 1, len(df_lookup))
    distances, indices = nn.kneighbors(embedding, n_neighbors=n_request)

    results = df_lookup.iloc[indices[0]][["channel_name", "channel_id"]].copy()
    results["similarity_score"] = distances[0]

    # Filter out the query channel itself
    results = results[results["channel_id"] != channel_id]

    # Return exactly top_k results
    results = results.head(top_k)

    return results.to_dict(orient="records")
