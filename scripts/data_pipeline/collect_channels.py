import json
import sys
from pathlib import Path
import boto3
from dotenv import load_dotenv
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv()

from src.config import settings
from src.youtube.client import yt_client

s3 = boto3.client("s3", region_name="us-west-2")

with open(PROJECT_ROOT / "data" / "consts" / "yt_api_queries.json", "r") as f:
    QUERIES = json.load(f)


def _search_channels(query: str, seen: set) -> list:
    collected = []
    try:
        results = yt_client.search_channels(query)
        for item in results:
            cid = item["channel_id"]
            if cid not in seen:
                seen.add(cid)
                collected.append(item)
    except Exception as e:
        print(f"search error for '{query}': {e}")
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


def collect():
    seen = set()
    total = 0

    for query in QUERIES:
        print(f"searching: {query}")
        candidates = _search_channels(query, seen)
        if not candidates:
            continue

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
            _save_to_s3(clean, query)
            total += len(clean)

    print(f"\ndone. collected {total} new channels total.")


def _make_safe_filename(query: str, ext: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_query = query.replace(" ", "_")[:30]
    return f"{timestamp}_{safe_query}.{ext}"


if __name__ == "__main__":
    collect()


"""
# Use this below to run the script:

python scripts/data_pipeline/collect_channels.py

"""