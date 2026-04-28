import logging

from src.config import settings
from src.youtube.client import yt_client

logger = logging.getLogger(__name__)


def _get_channel_data(channel_name: str) -> dict | None:
    results = yt_client.search_channels(channel_name, max_results=1)

    if not results:
        logger.warning("Could not find channel: %s", channel_name)
        return None

    channel = results[0]
    channel_id = channel["channel_id"]
    logger.info("Found channel: '%s' (id: %s)", channel["channel_name"], channel_id)

    details = yt_client.get_channel_info(channel_id)
    if details is None:
        return None

    videos = yt_client.get_videos(details.get("uploads", ""), max_videos=settings.max_videos_per_channel)
    details["videos"] = videos
    details["channel_id"] = channel_id

    details.pop("uploads", None)
    return details