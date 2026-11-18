"""
Minimal Supabase client over its REST API (PostgREST), using only `requests`.

Supabase is optional: if no URL/key is configured, every call is a no-op and the app still
works. Images are never sent; a prediction row stores a SHA-256 hash of the image bytes,
the predicted class, the confidence and the full probability vector.
"""
from __future__ import annotations
import os

import requests


class SupabaseLog:
    def __init__(self, url: str | None = None, key: str | None = None, timeout: float = 8.0):
        self.url = (url or os.getenv("SUPABASE_URL") or "").rstrip("/")
        self.key = key or os.getenv("SUPABASE_KEY") or ""
        self.timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self.url and self.key)

    def _headers(self, extra: dict | None = None) -> dict:
        h = {"apikey": self.key, "Authorization": f"Bearer {self.key}",
             "Content-Type": "application/json"}
        h.update(extra or {})
        return h

    def insert(self, table: str, row: dict) -> None:
        """Insert one row. Raises requests.HTTPError on failure."""
        r = requests.post(f"{self.url}/rest/v1/{table}", json=row, timeout=self.timeout,
                          headers=self._headers({"Prefer": "return=minimal"}))
        r.raise_for_status()

    def select(self, table: str, params: dict | None = None) -> list[dict]:
        r = requests.get(f"{self.url}/rest/v1/{table}", params=params or {},
                         timeout=self.timeout, headers=self._headers())
        r.raise_for_status()
        return r.json()
