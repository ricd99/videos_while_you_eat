from googleapiclient.discovery import build
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("YOUTUBE_API_KEY_RYA")
yt = build("youtube", "v3", developerKey=API_KEY)

def get_channel_id_from_name(channel_name: str) -> str | None:
    resp = yt.search().list(
        part="snippet",
        q = channel_name,
        type="channel",
        max_Results=1
    ).execute()

    items = resp.get("items", [])
    if not items:
        print(f"could not find channel: {channel_name}")
        return None
    
    return items[0]["snippet"]["channelId"] #TODO: return options for user to choose

