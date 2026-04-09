import json
import boto3
import psycopg2
import pandas as pd
from dotenv import load_dotenv
import os
import sys
from pathlib import Path

load_dotenv()

from src.data.preprocess_data import preprocess_data
from src.features.build_features import build_features

s3 = boto3.client("s3", region_name="us-west-2")
BUCKET = "ytrec-data-lake"

conn = psycopg2.connect(
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
        body = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
        channels = json.loads(body)
        all_channels.extend(channels)

    print(f"loaded {len(all_channels)} total channels from S3")
    return all_channels

def _insert_into_rds(conn, df: pd.DataFrame, table: str, columns: list[str]):
    pass #TODO: function to abstract repeated code from insert functions below


def _insert_cleaned(conn, df: pd.DataFrame):
    cur = conn.cursor()
    inserted = 0
    for _, row in df.iterrows():
        cur.execute("""
            INSERT INTO channels_cleaned (channel_id, channel_name, description, topics, keywords, videos)
            VALUES = (%s, %s, %s, %s, %s, %s)
        """, (
            row["channel_id"],
            row["channel_name"],
            str(row.get("description")),
            str(row.get("topics", "")),
            str(row.get("keywords", "")),
            str(row.get("videos", "")),
        ))
        inserted += cur.rowcount
    conn.commit()
    cur.close()
    print(f"inserted {inserted} new rows into channels_cleaned")

def _insert_features(conn, df: pd.DataFrame):
    cur = conn.cursor()
    inserted = 0
    for _, row in df.iterrows():
        cur.execute("""
            INSERT INTO channels_cleaned (channel_id, channel_name, description, topics, keywords, videos)
            VALUES = (%s, %s, %s, %s, %s, %s)
        """, (
            row["channel_id"],
            row["channel_name"],
            str(row.get("description")),
            str(row.get("topics", "")),
            str(row.get("keywords", "")),
            str(row.get("videos", "")),
        ))
        inserted += cur.rowcount
    conn.commit()
    cur.close()
    print(f"inserted {inserted} new rows into channels_cleaned")