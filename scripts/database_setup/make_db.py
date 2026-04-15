import psycopg2
from dotenv import load_dotenv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv()

from src.config import settings

conn = psycopg2.connect(
    host=settings.rds_host,
    port=5432,
    database="postgres",
    user="postgres",
    password=settings.rds_password,
    sslmode="require"
)

cur = conn.cursor()

cur.execute("""
    CREATE TABLE IF NOT EXISTS channels_cleaned (
        channel_id TEXT PRIMARY KEY,
        channel_name TEXT,
        description TEXT,
        topics TEXT,
        keywords TEXT,
        videos TEXT
    );
""")

cur.execute("""
    CREATE TABLE IF NOT EXISTS channels_final (
        channel_id TEXT PRIMARY KEY,
        channel_name TEXT,
        text TEXT
    );
""")

conn.commit()
cur.close()
conn.close()