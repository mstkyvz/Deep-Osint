"""Shared HTTP session with retries and an on-disk cache."""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import requests

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/140.0 Safari/537.36"
)

_session: requests.Session | None = None


def session() -> requests.Session:
    global _session
    if _session is None:
        s = requests.Session()
        s.headers["User-Agent"] = os.environ.get("DEEPOSINT_UA", UA)
        proxy = os.environ.get("DEEPOSINT_PROXY")
        if proxy:
            s.proxies = {"http": proxy, "https": proxy}
        _session = s
    return _session


def get(url: str, *, cache_dir: Path | None = None, timeout: float = 30, retries: int = 3,
        method: str = "GET", data: dict | None = None) -> bytes:
    """Fetch url (GET or POST form) with simple retry/backoff; cache body by url+data hash."""
    cache_file = None
    if cache_dir is not None:
        key = hashlib.sha1((method + url + repr(sorted((data or {}).items()))).encode()).hexdigest()
        cache_file = cache_dir / key[:2] / key
        if cache_file.exists() and cache_file.stat().st_size > 0:
            return cache_file.read_bytes()
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = session().request(method, url, data=data, timeout=timeout)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"{r.status_code} for {url}")
            r.raise_for_status()
            body = r.content
            if cache_file is not None:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_bytes(body)
            return body
        except requests.RequestException as e:  # noqa: PERF203
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"request failed after {retries} tries: {url}") from last
