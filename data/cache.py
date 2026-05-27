"""SQLite-backed cache for scraped/API data.

Single table keyed by string; values are pickled. Each call opens a fresh
connection so the module is safe to use from Streamlit's thread pool.

Usage:
    from data.cache import cache_get, cache_set, cached
    from config import TTL_FBREF

    df = cache_get("fbref:big5:2024-25", TTL_FBREF)
    if df is None:
        df = scrape()
        cache_set("fbref:big5:2024-25", df)

    @cached("elo:world", TTL_ELO)
    def fetch_elo(): ...
"""
from __future__ import annotations

import pickle
import sqlite3
import time
from functools import wraps
from typing import Any, Callable

from config import CACHE_DIR

_DB_PATH = CACHE_DIR / "cache.sqlite"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH, timeout=10.0)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS cache ("
        " key TEXT PRIMARY KEY,"
        " value BLOB NOT NULL,"
        " ts REAL NOT NULL"
        ")"
    )
    return conn


def cache_get(key: str, ttl_seconds: float) -> Any | None:
    """Return cached value for key if present and within TTL, else None."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT value, ts FROM cache WHERE key = ?", (key,)
        ).fetchone()
    if row is None:
        return None
    value_blob, ts = row
    if time.time() - ts > ttl_seconds:
        return None
    return pickle.loads(value_blob)


def cache_set(key: str, value: Any) -> None:
    """Store value under key with the current timestamp."""
    blob = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    with _connect() as conn:
        conn.execute(
            "INSERT INTO cache (key, value, ts) VALUES (?, ?, ?)"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value, ts = excluded.ts",
            (key, blob, time.time()),
        )
        conn.commit()


def cache_age(key: str) -> float | None:
    """Seconds since key was written, or None if missing. Used by the UI to
    display a cached-data timestamp on scraper failure."""
    with _connect() as conn:
        row = conn.execute("SELECT ts FROM cache WHERE key = ?", (key,)).fetchone()
    return None if row is None else time.time() - row[0]


def clear(prefix: str | None = None) -> int:
    """Drop cache entries. With no prefix, clears everything; with a prefix,
    clears only matching keys. Returns the number of rows deleted.
    Wire this to the Streamlit 'Refresh data' button."""
    with _connect() as conn:
        if prefix is None:
            cur = conn.execute("DELETE FROM cache")
        else:
            cur = conn.execute("DELETE FROM cache WHERE key LIKE ?", (f"{prefix}%",))
        conn.commit()
        return cur.rowcount


def cached(key: str, ttl_seconds: float) -> Callable:
    """Decorator: cache the wrapped function's return value under a fixed key.

    For per-argument caching, pass arguments into the key from the caller:
        @cached(f"goalscorer:{match_id}", TTL_GOALSCORER_ODDS)
    """
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            hit = cache_get(key, ttl_seconds)
            if hit is not None:
                return hit
            result = fn(*args, **kwargs)
            cache_set(key, result)
            return result
        return wrapper
    return decorator
