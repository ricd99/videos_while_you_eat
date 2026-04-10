import json
import boto3
import psycopg2
import pandas as pd
from dotenv import load_dotenv
import os
from psycopg2.extras import execute_values
import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))
load_dotenv()

from src.data.preprocess_data import preprocess_data
from src.features.build_features import build_features
from src.data.fetch_data_given_query_channel import _get_video_features

s3 = boto3.client("s3", region_name="us-west-2")
BUCKET = "ytrec-data-lake"

def _get_db_connection():
    return psycopg2.connect(
        host=os.getenv("RDS_HOST"),
        port=5432,
        database="postgres",
        user="postgres",
        password=os.getenv("RDS_PW"),
        sslmode="require"
    )

def _get_existing_channel_ids(conn) -> set:
    cur = conn.cursor()
    cur.execute("SELECT channel_id FROM channels_cleaned")
    ids = {row[0] for row in cur.fetchall()}
    cur.close()
    return ids

def _load_raw_from_s3() -> list:
    all_channels = []
    response = s3.list_objects_v2(Bucket=BUCKET, Prefix="raw/")

    for obj in response.get("Contents", []):
        key = obj["Key"]
        if not key.endswith(".json"):
            continue
        body = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
        if not body:
            print(f"skipping empty file: {key}")
            continue
        channels = json.loads(body)
        all_channels.extend(channels)

    print(f"loaded {len(all_channels)} total channels from S3")
    return all_channels

def _insert_into_rds(conn, df: pd.DataFrame, table: str, columns: list[str]):
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

def _append_video_data(new_channels):                    # video data collected here
    for channel in new_channels:
        uploads = channel.get("uploads")
        if uploads:
            print(f"fetching videos for {channel.get("channel_name")}")
            channel["videos"] = _get_video_features(uploads)
        else:
            channel["videos"] = []


def run_etl():
    conn = _get_db_connection()
    existing_ids = _get_existing_channel_ids(conn)
    print(f"found {len(existing_ids)} existing channels in RDS")
    raw_channels = _load_raw_from_s3()

    new_channels = [c for c in raw_channels if c.get("channel_id") not in existing_ids] #TODO: faster ways other than one by one iteration (here and in general for this project)
    print(f"{len(new_channels)} new channels to process")

    if not new_channels:
        print("no new channels, no etl to do")
        conn.close()
        return
    
    try:
        _append_video_data(new_channels)
    except Exception as e:
        print(f"stopped fetching videos: {e}")


    df = pd.DataFrame(new_channels)
    df = preprocess_data(df)
    print(f"preprocessing s3 done")

    _insert_into_rds(conn, df, "channels_cleaned", ["channel_id", "channel_name", "description", "topics", "keywords", "videos"])
    df_features = build_features(df)
    _insert_into_rds(conn, df, "channels_features", ["channel_id", "channel_name", "text"])

    conn.close()
    print("ETL complete")

if __name__ == "__main__":
    run_etl()


"""
# Use this below to run the script:

python scripts/data_pipeline/etl.py

"""