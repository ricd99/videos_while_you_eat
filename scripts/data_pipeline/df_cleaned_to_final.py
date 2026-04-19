import pandas as pd
from dotenv import load_dotenv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv()

from src.db.connection import db_manager
from src.features.build_features import build_features


def _get_df_all_rows(table: str) -> pd.DataFrame:
    return db_manager.fetch_dataframe(f"SELECT * FROM {table}")


def run_script():
    df_cleaned = _get_df_all_rows("channels_cleaned")
    df_final = _get_df_all_rows("channels_final")

    existing_final_ids = set(df_final["channel_id"])
    df = df_cleaned[~df_cleaned["channel_id"].isin(existing_final_ids)]
    print(f"{len(df)} channels to process into channels_final")

    df = build_features(df)
    inserted = db_manager.insert_dataframe(
        df,
        "channels_final",
        ["channel_id", "channel_name", "text"]
    )
    print(f"inserted {inserted} new rows into channels_final")
    db_manager.close()


if __name__ == "__main__":
    run_script()


"""
# Use this below to run the script:

python scripts/data_pipeline/df_cleaned_to_final.py

"""