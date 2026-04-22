from pathlib import Path
from typing import Optional
import logging

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from sqlalchemy import create_engine

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FALLBACK_CSV = PROJECT_ROOT / "data" / "processed" / "channels_pp.csv"


class DatabaseManager:
    def __init__(self):
        self._engine = None
        self._conn = None

    def connect(self):
        from src.config import settings
        return psycopg2.connect(
            host=settings.rds_host,
            port=5432,
            database="postgres",
            user="postgres",
            password=settings.rds_password,
            sslmode="require"
        )

    def create_engine(self):
        from src.config import settings
        return create_engine(settings.rds_url)

    def get_connection(self) -> psycopg2.connect:
        if self._conn is None or self._conn.closed:
            self._conn = self.connect()
        return self._conn

    def get_engine(self):
        if self._engine is None:
            self._engine = self.create_engine()
        return self._engine

    def execute(self, query: str, params: tuple = None):
        conn = self.get_connection()
        cur = conn.cursor()
        if params:
            cur.execute(query, params)
        else:
            cur.execute(query)
        conn.commit()
        cur.close()

    def fetch_all(self, query: str) -> list:
        conn = self.get_connection()
        cur = conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
        cur.close()
        return rows

    def fetch_dataframe(self, query: str) -> pd.DataFrame:
        engine = self.get_engine()
        return pd.read_sql(query, engine)

    def insert_dataframe(
        self,
        df: pd.DataFrame,
        table: str,
        columns: list[str]
    ) -> int:
        conn = self.get_connection()
        cur = conn.cursor()
        col_names = ", ".join(columns)
        rows = [tuple(row.get(col) for col in columns) for row in df.to_dict("records")]
        execute_values(
            cur,
            f"INSERT INTO {table} ({col_names}) VALUES %s ON CONFLICT (channel_id) DO NOTHING",
            rows
        )
        inserted = cur.rowcount
        conn.commit()
        cur.close()
        return inserted

    def table_exists(self, table_name: str) -> bool:
        conn = self.get_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = %s)",
            (table_name,)
        )
        exists = cur.fetchone()[0]
        cur.close()
        return exists

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None
        if self._engine:
            self._engine.dispose()
            self._engine = None

    def load_lookup_table(self, timeout: int = 3) -> Optional[pd.DataFrame]:
        try:
            conn = self.get_connection()
            conn.timeout = timeout  # Set connection timeout
            return self.fetch_dataframe("SELECT channel_id, channel_name FROM channels_cleaned")
        except Exception:
            return None

    def load_lookup_table_with_fallback(self) -> pd.DataFrame:
        from src.config import settings

        # In production, try DB first (set use_database=true in .env)
        if settings.use_database:
            try:
                db_lookup = self.load_lookup_table(timeout=5)
                if db_lookup is not None and not db_lookup.empty:
                    return db_lookup
            except Exception as e:
                logger.warning("DB lookup failed: %s", e)

        # Default: use CSV (fast, no network dependency)
        if FALLBACK_CSV.exists():
            df = pd.read_csv(FALLBACK_CSV)
            return df[["channel_id", "channel_name"]]
        else:
            raise RuntimeError(
                "Database unavailable and no fallback CSV found. "
                f"Expected: {FALLBACK_CSV}"
            )


db_manager = DatabaseManager()