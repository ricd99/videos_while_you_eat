from googleapiclient.discovery import build
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("YOUTUBE_API_KEY_RYA")
yt = build("youtube", "v3", developerKey=API_KEY)