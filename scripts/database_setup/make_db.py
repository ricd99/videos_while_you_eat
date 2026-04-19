from dotenv import load_dotenv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv()

from src.db.connection import db_manager

conn = db_manager.get_connection()
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
db_manager.close()
print("Database tables created successfully.")