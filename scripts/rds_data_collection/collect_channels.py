import json
from pathlib import Path
import boto3
from googleapiclient.discovery import build
from dotenv import load_dotenv
import os
from datetime import datetime, timezone

"""
Script that collects new channels from yt api. All features except videos are collected here as videos is an expensive api call. 
Channels that don't meet filtering requirements (sub/video count, last publish date) have flags attached to their entry
"""

load_dotenv()

yt = build("youtube", "v3", developerKey=os.getenv("YOUTUBE_API_KEY_RYA"))
s3 = boto3.client("s3", region_name="us-west-2")

BUCKET = "ytrec-data-lake"
MIN_SUBSCRIBERS = 1000
MIN_VIDEOS = 10
MONTHS_INACTIVE = 6

with open(Path.cwd() / "src" / "data" / "queries.json", "r") as f:
    QUERIES = json.load(f)

def _search_channels(query: str, seen: set) -> list:
    collected = []
    try:
        resp = yt.search().list(
            part="snippet",
            q=query,
            type="channel",
            relevanceLanguage="en",
            maxResults=50,
        ).execute()

        for item in resp.get("items", []):
            cid = item["snippet"]["channelId"]
            if cid not in seen:
                seen.add(cid)
                collected.append({
                    "channel_id": cid,
                    "channel_name": item["snippet"]["title"],
                })
    except Exception as e:
        print(f"search error for '{query}': {e}")
    return collected

def _get_channel_details(channel_ids: list) -> list:
    results = []
    # batch up to 50 at a time — youtube api allows this
    for i in range(0, len(channel_ids), 50):
        batch = channel_ids[i:i+50]
        try:
            resp = yt.channels().list(
                part="snippet,statistics,contentDetails,topicDetails,brandingSettings",
                id=",".join(batch)
            ).execute()

            for ch in resp.get("items", []):
                snippet = ch.get("snippet", {})
                stats = ch.get("statistics", {})
                topic = ch.get("topicDetails", {})
                branding = ch.get("brandingSettings", {}).get("channel", {})
                uploads = ch.get("contentDetails", {}).get("relatedPlaylists", {}).get("uploads")

                # check last upload date
                months_since_publish = None
                if published_at:
                    published = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                    months_since_publish = (datetime.now(timezone.utc) - published).days / 30

                # filter inactive channels
                published_at = snippet.get("publishedAt", "")
                subscriber_count = int(stats.get("subscriberCount", 0))
                video_count = int(stats.get("videoCount", 0))

                flags = []
                if subscriber_count < MIN_SUBSCRIBERS:
                    flags.append("low_subscribers")
                if video_count < MIN_VIDEOS:
                    flags.append("low_videos")

                results.append({
                    "channel_id": ch["id"],
                    "channel_name": snippet.get("title"),
                    "description": snippet.get("description"),
                    "country": snippet.get("country"),
                    "topics": topic.get("topicCategories"),
                    "keywords": branding.get("keywords"),
                    "uploads": uploads,
                    "subscriber_count": subscriber_count,
                    "video_count": video_count,
                    "months_since_publish": months_since_publish,
                    "flags": flags,
                })
        except Exception as e:
            print(f"channel details error: {e}")
    return results

def _save_to_s3(data: list, query: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_query = query.replace(" ", "_")[:30]
    key = f"raw/{timestamp}_{safe_query}.json"

    s3.put_object(
        Bucket=BUCKET,
        Key=key,
        Body=json.dumps(data, indent=2),
        ContentType="application/json"
    )
    print(f"saved {len(data)} channels to s3://{BUCKET}/{key}")

def collect():
    seen = set()
    total = 0

    for query in QUERIES:
        print(f"searching: {query}")
        candidates = _search_channels(query, seen)
        if not candidates:
            continue

        channel_ids = [c["channel_id"] for c in candidates]
        detailed = _get_channel_details(channel_ids)

        # filter out low quality channels before saving
        clean = [c for c in detailed if not any(f in c["flags"] for f in ["low_subscribers", "low_videos"])]
        flagged = [c for c in detailed if c not in clean]

        print(f"  found {len(detailed)} channels, {len(clean)} passed filters, {len(flagged)} flagged")

        if clean:
            _save_to_s3(clean, query)
            total += len(clean)

    print(f"\ndone. collected {total} new channels total.")


if __name__ == "__main__":
    collect()

