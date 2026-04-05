from googleapiclient.discovery import build
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("YOUTUBE_API_KEY_RYA")
yt = build("youtube", "v3", developerKey=API_KEY)

def _get_channel_id_from_name(channel_name: str) -> str | None:
    resp = yt.search().list(
        part="snippet",
        q = channel_name,
        type="channel",
        max_Results=1
    ).execute()

    items = resp.get("items", [])
    if not resp.get("items"):
        print(f"could not find channel: {channel_name}")
        return None
    
    return items[0]["snippet"]["channelId"] #TODO: return options for user to choose

def _get_channel_features(channel_id: str) -> dict | None:
    resp = yt.channels().list(
        part="brandingSettings,contentDetails,snippet,topicDetails",
        id=channel_id
    ).execute()

    items = resp.get("items", [])
    if not resp.get("items"):
        print(f"could not find channel: {channel_id}")
        return None
    
    ch = resp["items"][0]
    snippet = ch.get("snippet", {})
    topic = ch.get("topicDetails", {})
    branding = ch.get("brandingSettings", {}).get("channel", {})
    uploads_playlist = ch.get("contentDetails", {}).get("relatedPlaylists", {})

    return {
        "channel_name": snippet.get("title"),
        "title":        snippet.get("title"),
        "description":  snippet.get("description"),
        "country":      snippet.get("country"),
        "topics":       topic.get("topicCategories"),
        "keywords":     branding.get("keywords"),
        "uploads":      uploads_playlist.get("uploads"),
    }

def _get_video_features(uploads: str, max_videos: int = 10) -> list:
    videos = []
    next_page = None

    while len(videos) < max_videos:
        resp = yt.playlistItems().list(
            part="snippet",
            playlistId=uploads,
            maxResults=50,
            pageToken=next_page
        ).execute()
    
    for item in resp["items"]:
        if len(videos) >= max_videos:
            break

        snippet = item["snippet"]
        description = snippet.get("description", "")
        if "#shorts" in description.lower():
            continue

        videos.append({
            "title":       snippet.get("title"),
            "description": description,
        })

        next_page = resp.get("nextPageToken")
        if not next_page:
            break

    return videos

def get_channel_data(channel_name: str) -> dict | None:
    channel_id = _get_channel_id_from_name(channel_name)
    if channel_id is None:
        return None

    features = _get_channel_features(channel_id)
    if features is None:
        return None
    
    videos = _get_video_features(features["uploads"])
    features["videos"] = videos
    features["channel_id"] = channel_id

    return features