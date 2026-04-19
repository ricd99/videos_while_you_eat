import joblib
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv()

from src.config import settings
from src.db.connection import db_manager
from src.data.fetch_data_given_query_channel import get_channel_data
from src.data.preprocess_data import preprocess_data
from src.features.build_features import build_features

ct = joblib.load(PROJECT_ROOT / "artifacts" / "column_transformer.pkl")
nn = joblib.load(PROJECT_ROOT / "artifacts" / "nn_model.pkl")


class LookupTable:
    def __init__(self):
        self._df = None

    def get(self) -> pd.DataFrame:
        if self._df is None:
            self._df = db_manager.load_lookup_table_with_fallback()
        return self._df


_lookup_table = LookupTable()


def predict(channel_name: str) -> list[dict] | dict:
    print(f"[DEBUG] Starting predict for: {channel_name}")
    t0 = time.time()

    input_dict = get_channel_data(channel_name)
    print(f"[DEBUG] Got channel data: {time.time() - t0:.2f}s")
    if input_dict is None:
        return {"error": f"could not find channel: {channel_name}"}

    df = pd.DataFrame([input_dict])
    print(f"[DEBUG] Created DataFrame: {time.time() - t0:.2f}s")

    df = preprocess_data(df)
    print(f"[DEBUG] Preprocessed: {time.time() - t0:.2f}s")

    df = build_features(df)
    print(f"[DEBUG] Built features: {time.time() - t0:.2f}s")

    print(f"[DEBUG] Columns before transform: {list(df.columns)}")

    df_transformed = ct.transform(df)
    print(f"[DEBUG] Transformed: {time.time() - t0:.2f}s")

    distances, indices = nn.kneighbors(df_transformed)
    print(f"[DEBUG] NN predict: {time.time() - t0:.2f}s")

    print(f"[DEBUG] Getting lookup table...")
    df_lookup = _lookup_table.get()
    print(f"[DEBUG] Got lookup table: {len(df_lookup)} rows")

    results = df_lookup.iloc[indices[0]][["channel_name", "channel_id"]].copy()
    results["similarity_score"] = distances[0]
    print(f"[DEBUG] Done: {time.time() - t0:.2f}s")
    return results.to_dict(orient="records")