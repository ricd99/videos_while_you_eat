import joblib
import psycopg2
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import os

from src.data.youtube_api import get_channel_data
from src.data.preprocess_data import preprocess_data
from src.features.build_features import build_features

load_dotenv()

ct = joblib.load(Path.cwd() / "src" / "serving" / "model" / "eacd9855d8444a0fad5bd82d2629fb78" / "artifacts" / "column_transformer.pkl") # works only if ran from project root
nn = joblib.load(Path.cwd() / "src" / "serving" / "model" / "eacd9855d8444a0fad5bd82d2629fb78" / "artifacts" / "nn_model.pkl")           #TODO use __file__ so can run from anywhere?

def _get_db_connection():
    return psycopg2.connect(
        host=os.getenv("RDS_HOST")
    )

def predict(channel_name: str) -> list[dict] | dict:
    input_dict = get_channel_data(channel_name)
    if input_dict is None:
        return {"error": f"could not find channel: {channel_name}"}
    
    df = pd.DataFrame([input_dict])
    
    df = preprocess_data(df)
    df = build_features(df)
    df_transformed = ct.transform(df)
    
    distances, indices = nn.kneighbors(df_transformed)
    
    results = df_all.iloc[indices[0]][["channel_name", "channel_id"]].copy()
    results["similarity_score"] = distances[0]
    return results.to_dict(orient="records")