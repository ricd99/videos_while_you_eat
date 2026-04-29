import json
import boto3
import pandas as pd
from dotenv import load_dotenv
import sys
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv()

from src.config import settings
from src.db.connection import db_manager
from src.data.preprocess_data import preprocess_data
from src.features.build_features import build_features
from src.data.fetch_data_given_query_channel import _get_channel_data, _get_channel_videos

s3 = boto3.client("s3", region_name="us-west-2")


def _get_existing_channel_ids() -> set:
    rows = db_manager.fetch_all("SELECT channel_id FROM channels_cleaned")
    return {row[0] for row in rows}


def _load_raw_from_s3() -> list:
    all_channels = []
    response = s3.list_objects_v2(Bucket=settings.s3_bucket, Prefix="raw/")

    for obj in response.get("Contents", []):
        key = obj["Key"]
        if not key.endswith(".json"):
            continue
        body = s3.get_object(Bucket=settings.s3_bucket, Key=key)["Body"].read()
        if not body:
            print(f"skipping empty file: {key}")
            continue
        channels = json.loads(body)
        all_channels.extend(channels)

    return all_channels


CHECKPOINT_FILE = PROJECT_ROOT / "data" / "misc" / "etl_checkpoint.json"


def _save_checkpoint(channel_id: str):
    os.makedirs(CHECKPOINT_FILE.parent, exist_ok=True) 
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"last_channel_id": channel_id}, f)


def _load_checkpoint() -> str | None:
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f).get("last_channel_id")
    return None


def run_etl() -> int:
    existing_ids = _get_existing_channel_ids()
    print(f"found {len(existing_ids)} existing channels in RDS")
    raw_channels = _load_raw_from_s3()
    print(f"loaded {len(raw_channels)} total channels from S3")
    new_channels = [c for c in raw_channels if c.get("channel_id") not in existing_ids]
    print(f"{len(new_channels)} new channels to process")

    if not new_channels:
        print("no new channels, no etl to do")
        return 0

    completed = []

    last_id = _load_checkpoint()
    if last_id and last_id in [c["channel_id"] for c in new_channels]:
        idx = [c["channel_id"] for c in new_channels].index(last_id)
        new_channels = new_channels[idx:]
        print(f"resuming from checkpoint: {last_id}")

    for channel in new_channels:
        try:
            print(f"appending video data for channel: {channel['channel_name']}")
            _get_channel_videos(channel)
            completed.append(channel)
        except Exception as e:
            break

    if completed:
        df = pd.DataFrame(completed)
        df = preprocess_data(df)
        print(f"preprocessing s3 done")

        db_manager.insert_dataframe(
            df,
            "channels_cleaned",
            ["channel_id", "channel_name", "description", "topics", "keywords", "videos"]
        )
        df = build_features(df)
        db_manager.insert_dataframe(
            df,
            "channels_final",
            ["channel_id", "channel_name", "text"]
        )
        _save_checkpoint(completed[-1]["channel_id"])

    db_manager.close()
    return len(completed)


if __name__ == "__main__":
    run_etl()


"""
# Use this below to run the script:

python scripts/data_pipeline/etl.py

"""