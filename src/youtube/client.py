import logging
from typing import Optional

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from src.config import settings

logger = logging.getLogger(__name__)

class AllAPIKeysExhaustedError(Exception):
    """Raised when all YouTube API keys have exceeded quota."""
    pass

class YouTubeClient:
    def __init__(self):
        self._yt = None
        self._build_client()

    def _build_client(self):
        key = settings.current_api_key
        self._yt = build(
            "youtube",
            "v3",
            developerKey=key,
            cache_discovery=False
        )

    def _rotate_key(self):
        if settings.rotate_api_key():
            logger.info("Rotating YouTube API key")
            self._build_client()
            return True
        return False

    def _try_execute(self, request, max_retries: int = 3):
        retries = 0
        while retries < max_retries:
            try:
                return request.execute()
            except HttpError as e:
                error_msg = str(e)
                if "quotaExceeded" in error_msg or "dailyLimitExceeded" in error_msg:
                    logger.warning("YouTube API quota exceeded, attempting key rotation")
                    if self._rotate_key():
                        retries += 1
                        request = request
                        continue
                    else:
                        logger.error("All API keys exhausted")
                        raise AllAPIKeysExhaustedError("all API keys have exceeded quota")
                raise
        raise RuntimeError("Max retries exceeded")

    def search_channels(self, query: str, max_results: int = 10) -> list[dict]:
        request = self._yt.search().list(
            part="snippet",
            q=query,
            type="channel",
            relevanceLanguage="en",
            maxResults=max_results
        )
        response = self._try_execute(request)

        results = []
        for item in response.get("items", []):
            results.append({
                "channel_id": item["snippet"]["channelId"],
                "channel_name": item["snippet"]["title"],
            })
        return results

    def get_channel_details(self, channel_ids: list[dict]) -> list[dict]:
        results = []
        for i in range(0, len(channel_ids), 50):
            batch = [c["channel_id"] for c in channel_ids[i:i+50]]
            request = self._yt.channels().list(
                part="snippet,statistics,contentDetails,topicDetails,brandingSettings",
                id=",".join(batch)
            )
            response = self._try_execute(request)

            for ch in response.get("items", []):
                snippet = ch.get("snippet", {})
                stats = ch.get("statistics", {})
                topic = ch.get("topicDetails", {})
                branding = ch.get("brandingSettings", {}).get("channel", {})

                results.append({
                    "channel_id": ch["id"],
                    "channel_name": snippet.get("title"),
                    "description": snippet.get("description"),
                    "country": snippet.get("country"),
                    "topics": topic.get("topicCategories"),
                    "keywords": branding.get("keywords"),
                    "uploads": ch.get("contentDetails", {}).get("relatedPlaylists", {}).get("uploads"),
                    "subscriber_count": int(stats.get("subscriberCount", 0)),
                    "video_count": int(stats.get("videoCount", 0)),
                })
        return results

    def get_channel_info(self, channel_id: str) -> Optional[dict]:
        request = self._yt.channels().list(
            part="brandingSettings,contentDetails,snippet,topicDetails",
            id=channel_id
        )
        try:
            response = self._try_execute(request)
        except Exception:
            return None

        items = response.get("items", [])
        if not items:
            return None

        ch = items[0]
        snippet = ch.get("snippet", {})
        topic = ch.get("topicDetails", {})
        branding = ch.get("brandingSettings", {}).get("channel", {})
        uploads = ch.get("contentDetails", {}).get("relatedPlaylists", {})

        return {
            "channel_name": snippet.get("title"),
            "description": snippet.get("description"),
            "country": snippet.get("country"),
            "topics": topic.get("topicCategories"),
            "keywords": branding.get("keywords"),
            "uploads": uploads.get("uploads"),
        }

    def get_videos(self, playlist_id: str, max_videos: int = 10) -> list[dict]:
        videos = []
        next_page = None
        pages_fetched = 0

        while len(videos) < max_videos and pages_fetched < 5:
            request = self._yt.playlistItems().list(
                part="snippet",
                playlistId=playlist_id,
                maxResults=50,
                pageToken=next_page if next_page else ""
            )
            try:
                response = self._try_execute(request)
            except Exception:
                break

            for item in response.get("items", []):
                if len(videos) >= max_videos:
                    break

                snippet = item["snippet"]
                desc = snippet.get("description", "")
                if "#shorts" in desc.lower():
                    continue

                videos.append({
                    "title": snippet.get("title"),
                    "description": desc,
                })

                next_page = response.get("nextPageToken")
                if not next_page:
                    break

            pages_fetched += 1

        return videos
    


yt_client = YouTubeClient() #singleton. if imported multiple times, still works, as python caches client.py module after first run.