import joblib
from sqlalchemy import create_engine
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv()

from src.data.fetch_data_given_query_channel import get_channel_data
from src.data.preprocess_data import preprocess_data
from src.features.build_features import build_features

ct = joblib.load(PROJECT_ROOT / "artifacts" / "column_transformer.pkl")
nn = joblib.load(PROJECT_ROOT / "artifacts" / "nn_model.pkl")

def _get_db_connection():
    host=os.getenv("RDS_HOST")
    password=os.getenv("RDS_PW")
    return create_engine(f"postgresql+psycopg2://postgres:{password}@{host}:5432/postgres?sslmode=require")

def _load_lookup_table() -> pd.DataFrame:
    engine = _get_db_connection()
    df = pd.read_sql("SELECT channel_id, channel_name FROM channels_cleaned;", engine)
    return df

df_lookup = _load_lookup_table()

def predict(channel_name: str) -> list[dict] | dict:

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