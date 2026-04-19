import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv()

from src.db.connection import db_manager
from src.data.preprocess_data import preprocess_data
from src.features.build_features import build_features

df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "channels_pp.csv")

df = preprocess_data(df)

inserted = db_manager.insert_dataframe(
    df,
    "channels_cleaned",
    ["channel_id", "channel_name", "description", "topics", "keywords", "videos"]
)
print(f"Inserted {inserted} rows into channels_cleaned")

df = build_features(df)
inserted = db_manager.insert_dataframe(
    df,
    "channels_final",
    ["channel_id", "channel_name", "text"]
)
print(f"Inserted {inserted} rows into channels_final")

db_manager.close()
print("Migration complete.")