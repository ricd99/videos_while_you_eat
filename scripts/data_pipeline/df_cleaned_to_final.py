import psycopg2
import pandas as pd
from dotenv import load_dotenv
import os
from psycopg2.extras import execute_values
import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))
load_dotenv()

from src.features.build_features import build_features

def _get_db_connection():
    return psycopg2.connect(
        host=os.getenv("RDS_HOST"),
        port=5432,
        database="postgres",
        user="postgres",
        password=os.getenv("RDS_PW"),
        sslmode="require"
    )

def _get_df_all_rows(conn, db) -> pd.DataFrame:
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM {db}")
    rows = cur.fetchall()
    colnames = [desc[0] for desc in cur.description]
    df = pd.DataFrame(rows, columns=colnames)
    cur.close()
    return df

def _insert_into_rds(conn, df: pd.DataFrame, table: str, columns: list[str]):            #TODO: abstract away, repeated function from etl.py
    cur = conn.cursor()
    col_names = ", ".join(columns)

    rows = [tuple(row.get(col) for col in columns) for row in df.to_dict("records")]

    execute_values(cur,
        f"INSERT INTO {table} ({col_names}) VALUES %s ON CONFLICT (channel_id) DO NOTHING",
        rows
    )

    inserted = cur.rowcount
    conn.commit()
    cur.close()
    print(f"inserted {inserted} new rows into {table}")

def run_script():
    conn = _get_db_connection()
    df_cleaned = _get_df_all_rows(conn, "channels_cleaned")
    df_final = _get_df_all_rows(conn, "channels_final")

    existing_final_ids = set(df_final["channel_id"])
    df = df_cleaned[~df_cleaned["channel_id"].isin(existing_final_ids)]
    print(f"{len(df)} channels to process into channels_final")

    df = build_features(df)
    _insert_into_rds(conn, df, "channels_final", ["channel_id", "channel_name", "text"])

    conn.close()

if __name__ == "__main__":
    run_script()


"""
# Use this below to run the script:

python scripts/data_pipeline/df_cleaned_to_final.py

"""
