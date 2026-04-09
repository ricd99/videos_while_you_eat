import psycopg2
import pandas as pd
from pathlib import Path
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
df = pd.read_csv(Path.cwd() / "data" / "processed" / "channels_pp.csv")

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
