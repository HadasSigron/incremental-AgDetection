"""SQLite-backed cache for incremental evaluation."""
from __future__ import annotations
import sqlite3
import json
import time
from pathlib import Path
from typing import Any, Optional

class CacheStore:
    """
    Tiny key-value store:
    - key: unique hash for (image + model + config)
    - value: serialized prediction (JSON string)
    Backed by a single SQLite file on disk.
    """
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Create the table if it does not exist and set a safe journal mode."""
        with sqlite3.connect(self.db_path) as db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)
            db.execute("PRAGMA journal_mode=WAL")

    def get(self, key: str) -> Optional[Any]:
        """Fetch a prediction by key; return parsed JSON or None if missing."""
        with sqlite3.connect(self.db_path) as db:
            cur = db.execute("SELECT value FROM cache WHERE key=?", (key,))
            row = cur.fetchone()
            return json.loads(row[0]) if row else None

    def put(self, key: str, value: Any) -> None:
        """Insert or update a prediction for key."""
        payload = json.dumps(value, ensure_ascii=False)
        ts = time.time()
        with sqlite3.connect(self.db_path) as db:
            db.execute(
                "REPLACE INTO cache(key, value, updated_at) VALUES (?, ?, ?)",
                (key, payload, ts),
            )
