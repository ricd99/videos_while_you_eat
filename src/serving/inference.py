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

# Lazy loading - loaded on first request, not at import
_nn = None
_embeddings = None
_lookup_df = None

def _load_artifacts():
    """Load model artifacts on first request."""
    global _nn, _embeddings, _lookup_df
    
    if _nn is None:
        artifacts = PROJECT_ROOT / "artifacts"
        
        # Load with error handling
        try:
            _nn = joblib.load(artifacts / "nn_model.pkl")
        except FileNotFoundError:
            raise RuntimeError("Model not found. Run training pipeline first.")
        
        try:
            _embeddings = joblib.load(artifacts / "embeddings.pkl")
        except FileNotFoundError:
            raise RuntimeError("Embeddings not found. Run training pipeline first.")
        
        try:
            _lookup_df = joblib.load(artifacts / "df_lookup.pkl")
        except FileNotFoundError:
            raise RuntimeError("Lookup not found. Run training pipeline first.")
    
    return _nn, _embeddings, _lookup_df



def predict(channel_name: str) -> list[dict] | dict:
    nn, embeddings, df_lookup = _load_artifacts()

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