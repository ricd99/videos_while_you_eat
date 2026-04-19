import joblib
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv()

from src.config import settings
from src.db.connection import db_manager
from src.data.fetch_data_given_query_channel import get_channel_data
from src.data.preprocess_data import preprocess_data
from src.features.build_features import build_features

ct = joblib.load(PROJECT_ROOT / "artifacts" / "column_transformer.pkl")
nn = joblib.load(PROJECT_ROOT / "artifacts" / "nn_model.pkl")


class LookupTable: # lazy loading db
    def __init__(self):
        self._df = None

    def get(self) -> pd.DataFrame:
        if self._df is None:
            self._df = db_manager.load_lookup_table_with_fallback()
        return self._df


_lookup_table = LookupTable()


def predict(channel_name: str) -> list[dict] | dict:
    input_dict = get_channel_data(channel_name)
    if input_dict is None:
        return {"error": f"could not find channel: {channel_name}"}

    df = pd.DataFrame([input_dict])
    df = preprocess_data(df)
    df = build_features(df)
    df_transformed = ct.transform(df)

    distances, indices = nn.kneighbors(df_transformed)

    df_lookup = _lookup_table.get()
    results = df_lookup.iloc[indices[0]][["channel_name", "channel_id"]].copy()
    results["similarity_score"] = distances[0]
    return results.to_dict(orient="records")