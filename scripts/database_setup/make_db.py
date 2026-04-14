import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

conn = psycopg2.connect(
    host=os.getenv("RDS_HOST"),
    port=5432,
    database="postgres",
    user="postgres",
    password=os.getenv("RDS_PW"),
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