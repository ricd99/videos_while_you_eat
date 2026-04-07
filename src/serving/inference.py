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
        host=os.getenv("RDS_HOST"),
        port=5432,
        database="postgres",
        user="postgres",
        password=os.getenv("RDS_PW"),
        sslmode="require"
    )

def _load_lookup_table() -> pd.DataFrame:
    conn = _get_db_connection()
    df = pd.read_sql("SELECT channel_id, channel_name FROM channels_cleaned;", conn)
    conn.close()
    return df


def predict(channel_name: str) -> list[dict] | dict:
    df_lookup = _load_lookup_table()

    input_dict = get_channel_data(channel_name)
    if input_dict is None:
        return {"error": f"could not find channel: {channel_name}"}
    
    df = pd.DataFrame([input_dict])
    df = preprocess_data(df)
    df = build_features(df)
    df_transformed = ct.transform(df)
    
    distances, indices = nn.kneighbors(df_transformed)
    
    results = df_lookup.iloc[indices[0]][["channel_name", "channel_id"]].copy()
    results["similarity_score"] = distances[0]
    return results.to_dict(orient="records")