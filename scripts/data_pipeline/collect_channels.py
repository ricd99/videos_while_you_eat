import json
import random
import sys
import argparse
from pathlib import Path
import boto3
from dotenv import load_dotenv
from datetime import datetime, timezone

QUERIES_PER_RUN = 50

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv()

from src.core.config import settings
from src.youtube.client import yt_client
from src.youtube.client import AllAPIKeysExhaustedError

s3 = boto3.client("s3", region_name="us-west-2")

with open(PROJECT_ROOT / "data" / "consts" / "yt_api_queries.json", "r") as f:
    QUERY_POOL = json.load(f)


def _select_queries_for_today() -> list:
    rng = random.Random(datetime.now().strftime("%Y-%m-%d"))
    selected = rng.sample(QUERY_POOL, min(QUERIES_PER_RUN, len(QUERY_POOL)))
    print(f"selected {len(selected)} queries for today's run")
    return selected


def _search_channels(query: str, seen: set) -> list:   # bubbles up errors raised by yt_client.search_channels to collect()
    collected = []
    results = yt_client.search_channels(query)
    for item in results:
        cid = item["channel_id"]
        if cid not in seen:
            seen.add(cid)
            collected.append(item)
    return collected

def _get_channel_details(channel_ids: list) -> list:
    return yt_client.get_channel_details(channel_ids)


def _save_to_s3(data: list, query: str):
    filename = _make_safe_filename(query, "json")

    s3.put_object(
        Bucket=settings.s3_bucket,
        Key=f"raw/{filename}",
        Body=json.dumps(data, indent=2),
        ContentType="application/json"
    )
    print(f"saved {len(data)} channels to s3://{settings.s3_bucket}/{filename}")


def collect(local: bool = False):
    seen = set()
    total = 0

    for query in _select_queries_for_today():
        try:
            print(f"searching: {query}")
            candidates = _search_channels(query, seen)
            if not candidates:
                continue
        except AllAPIKeysExhaustedError:
            print(f"all api keys exhausted")
            break 
        except Exception as e:
            print(f"search error for '{query}': {e}")

        channel_ids = [{"channel_id": c["channel_id"]} for c in candidates]
        detailed = _get_channel_details(channel_ids)

        filename = _make_safe_filename(query, "json")
        data_dir = PROJECT_ROOT / "data" / "raw" / "collected_channels"
        data_dir.mkdir(parents=True, exist_ok=True)
        data_file = data_dir / filename
        with open(data_file, "w") as f:
            json.dump(detailed, f, indent=2)

        # Filter by thresholds
        clean = [c for c in detailed if c.get("subscriber_count", 0) >= settings.min_subscribers and c.get("video_count", 0) >= settings.min_videos]
        flagged = [c for c in detailed if c not in clean]

        print(f"  found {len(detailed)} channels, {len(clean)} passed filters, {len(flagged)} flagged")

        if clean:
            if not local:
                try:
                    _save_to_s3(clean, query)
                except Exception as e:
                    print(f"s3 save failed for '{query}': {e}, skipping upload")
            total += len(clean)

    print(f"\ndone. collected {total} new channels total.")


def _make_safe_filename(query: str, ext: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_query = query.replace(" ", "_")[:30]
    return f"{timestamp}_{safe_query}.{ext}"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Collect channels from YouTube API")
    p.add_argument("--local", action="store_true", help="skip S3 upload (dev runs)")
    collect(local=p.parse_args().local)


"""
# Use this below to run the script:

python scripts/data_pipeline/collect_channels.py

"""