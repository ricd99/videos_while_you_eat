import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()
RDS_PW = os.getenv("RDS_PW")

conn = psycopg2.connect(
    host="ytrec-db-3.cx8wkqkwq9oo.us-west-2.rds.amazonaws.com",
    port=5432,
    database="postgres",
    user="postgres",
    password=RDS_PW,
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