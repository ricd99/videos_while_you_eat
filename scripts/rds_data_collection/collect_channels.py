import json
import boto3
from googleapiclient.discovery import build
from dotenv import load_dotenv
import os
from datetime import datetime, timezone

load_dotenv()

yt = build("youtube", "v3", developerKey=os.getenv("YOUTUBE_API_KEY"))
s3 = boto3.client("s3", region_name="us-west-2")