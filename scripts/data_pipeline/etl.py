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
    
    return all_channels