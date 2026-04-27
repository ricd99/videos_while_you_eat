from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
import os


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv() # load here so os.environ works (in _load_api_keys_) works?


class Settings(BaseSettings):
    rds_host: str = ""
    rds_password: str = ""
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    s3_bucket: str = "ytrec-data-lake"
    min_subscribers: int = 1000
    min_videos: int = 10
    months_inactive: int = 6
    max_videos_per_channel: int = 10
    max_pages_per_channel: int = 5
    model_run_id: Optional[str] = None
    use_database: bool = False  # Set to True in production to use RDS instead of CSV

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._key_index = 0
        self._youtube_api_keys: list[str] = []
        self._load_api_keys()

    def _load_api_keys(self):
        for key, value in os.environ.items():
            if key.startswith("YOUTUBE_API_KEY_") and value:
                self._youtube_api_keys.append(value)

    @property
    def youtube_api_keys(self) -> list[str]:
        return self._youtube_api_keys

    @property
    def current_api_key(self) -> str:
        if not self._youtube_api_keys:
            raise ValueError("No YouTube API keys configured")
        return self._youtube_api_keys[self._key_index]

    def rotate_api_key(self):
        if len(self._youtube_api_keys) > 1:
            self._key_index = (self._key_index + 1) % len(self._youtube_api_keys)
            return True
        return False

    @property
    def rds_url(self) -> str:
        return f"postgresql+psycopg2://postgres:{self.rds_password}@{self.rds_host}:5432/postgres?sslmode=require"


settings = Settings()
