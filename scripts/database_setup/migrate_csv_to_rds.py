import psycopg2
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
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
df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "channels_pp.csv")

for _, row in df.iterrows():
    cur.execute("""
        INSERT INTO channels_cleaned (channel_id, channel_name, description, topics, keywords, videos)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (channel_id) DO NOTHING;
    """, (
        row["channel_id"],
        row["channel_name"],
        row["description"],
        str(row["topics"]),
        str(row["keywords"]),
        str(row["videos"])
    ))

conn.commit()
cur.close()
conn.close()
