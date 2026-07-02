"""Content-addressed SQLite cache for the tool-call network seam (Phase 2).

This is the WAVE-0 CONTRACT for Phase 2 (see phase2_plan.md). It fixes the public
API — `Cache`, `norm_query`, `norm_url`, `CANNED`, `CacheMiss`, `MISS_POLICIES` —
so the parallel workstreams can build against stable signatures. WS-A owns
hardening (error-row caching, concurrency stress, unit tests) and wiring it into
`setup_tools`; the skeleton here is deliberately minimal but functional so
dependent workstreams can import and exercise it.

Design (locked in phase2_plan.md):
  - Wraps the RAW backend closures (build_search/build_scrape in backends.py)
    BEFORE tools.py applies MAX_TOOL_CHARS -- so the stored response is uncapped
    and the cap stays retunable without re-scraping.
  - Content-addressed: key_hash = sha256(namespace | normalized-key | version).
  - One cache, three miss policies, selected by a flag:
      live   -> on miss, call fn, store, return it        (SFT, product)
      canned -> on miss, DO NOT call fn; return a canned   (GRPO / frozen)
                constant and count the miss
      error  -> on miss, raise CacheMiss                   (strict debugging)
  - The cache is really a FROZEN FIXTURE DATASET with a capture date; its source
    of truth is S3 (WS-D syncs the sqlite file). `data/` is git-ignored.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import datetime, timezone
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

# Miss policies (see module docstring).
MISS_POLICIES = ("live", "canned", "error")

# Deterministic canned constants returned on a miss under the "canned" policy.
# These become part of the frozen training distribution -- keep them STABLE.
CANNED = {
    "search": "(no results)",
    "scrape": "(page not available)",
}

# Query params dropped during URL normalization (tracking noise that would
# otherwise fragment the scrape cache across identical pages). Bare "ref" is
# deliberately NOT here: on some sites it selects content (e.g. GitHub branch
# refs), and a fragmented cache (extra fetch) beats serving the wrong page.
_TRACKING_PARAMS = frozenset({
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid", "gclsrc", "dclid", "msclkid", "mc_cid", "mc_eid",
    "ref_src", "igshid",
})


class CacheMiss(Exception):
    """Raised on a cache miss under miss_policy='error'."""


# ---------------------------------------------------------------------------
# Key normalization -- importable so callers pass these as key_fn to wrap().
# ---------------------------------------------------------------------------
def norm_query(query: str) -> str:
    """Normalize a search query: lowercase, collapse whitespace."""
    return " ".join(query.lower().split())


def norm_url(url: str) -> str:
    """Canonicalize a URL: lowercase scheme+host, drop fragment + tracking
    params, strip a trailing slash on the path. Keeps meaningful query params."""
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower()
    netloc = parts.netloc.lower()
    path = parts.path.rstrip("/") or "/"
    # keep_blank_values: ?a=&b=1 must not silently collapse to ?b=1.
    kept = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if k.lower() not in _TRACKING_PARAMS
    ]
    query = urlencode(sorted(kept))
    return urlunsplit((scheme, netloc, path, query, ""))  # fragment dropped


def norm_scrape(url: str, mode: str = "direct") -> str:
    """Cache key for a scrape call: canonical URL + mode.

    The local backend's "direct" and "browser" modes return genuinely different
    content (browser waits for network-idle and auto-scrolls; direct's escalation
    is a no-scroll render), and the model's chosen mode is baked into the recorded
    trajectory -- so they MUST be distinct cache entries. Key on the *requested*
    mode; the stored value is whatever that mode ultimately returned (incl. a
    direct->browser escalation)."""
    return f"{norm_url(url)}\x00{mode}"


# ---------------------------------------------------------------------------
# Response -> status classification (negative caching).
# ---------------------------------------------------------------------------
def _default_status(response: str) -> str:
    return "ok" if (response or "").strip() else "empty"


# The local scrape backend never raises -- it returns these readable sentinels so
# one bad URL can't kill an episode. They're transient/failure signals, not menu
# content, so the cache marks them 'error': under miss_policy="live" an 'error'
# row is RE-FETCHED on the next populate pass (self-healing) instead of freezing a
# one-off timeout / rate-limit into the frozen corpus. Under "canned" the recorded
# string is still served verbatim (deterministic replay of what the agent saw).
_SCRAPE_FAILURE_MARKERS = ("(scrape failed", "(page returned no content)", "(page not available)")


def scrape_status(response: str) -> str:
    """Status for a scrape response: 'error' for the backend's failure sentinels,
    else 'empty'/'ok'. Pass as `status_fn=scrape_status` when wrapping scrape."""
    r = (response or "").strip()
    if not r:
        return "empty"
    if r.startswith(_SCRAPE_FAILURE_MARKERS):
        return "error"
    return "ok"


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
_SCHEMA = """
CREATE TABLE IF NOT EXISTS cache (
  key_hash      TEXT PRIMARY KEY,
  namespace     TEXT NOT NULL,
  key           TEXT NOT NULL,
  args_json     TEXT NOT NULL,
  response      TEXT,
  provider      TEXT,
  status        TEXT NOT NULL,     -- 'ok' | 'empty' | 'error'
  cache_version INTEGER NOT NULL,
  captured_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ns ON cache(namespace);
"""


class Cache:
    """A content-addressed SQLite cache with a pluggable miss policy.

    Args:
        path: sqlite file path (e.g. "data/cache.sqlite"); ":memory:" for tests.
        miss_policy: one of MISS_POLICIES.
        cache_version: bump whenever the stored response SHAPE changes; it is
            folded into the key hash so old rows never collide with new ones.
    """

    def __init__(self, path: str, *, miss_policy: str = "live", cache_version: int = 1):
        if miss_policy not in MISS_POLICIES:
            raise ValueError(f"miss_policy must be one of {MISS_POLICIES}, got {miss_policy!r}")
        self.path = path
        self.miss_policy = miss_policy
        self.cache_version = cache_version
        self._hits = self._misses = self._writes = 0
        self._lock = threading.Lock()
        # check_same_thread=False + WAL supports the parallel corpus build (WS-C).
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # -- key + row helpers --------------------------------------------------
    def _key_hash(self, namespace: str, key: str) -> str:
        raw = f"{namespace}\x00{key}\x00{self.cache_version}".encode()
        return hashlib.sha256(raw).hexdigest()

    def _get(self, namespace: str, key: str) -> sqlite3.Row | None:
        cur = self._conn.execute(
            "SELECT * FROM cache WHERE key_hash = ?", (self._key_hash(namespace, key),)
        )
        return cur.fetchone()

    def _set(self, namespace, key, args, response, provider, status):
        row = (
            self._key_hash(namespace, key), namespace, key, json.dumps(args),
            response, provider, status, self.cache_version,
            datetime.now(timezone.utc).isoformat(),
        )
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO cache "
                "(key_hash, namespace, key, args_json, response, provider, status, "
                " cache_version, captured_at) VALUES (?,?,?,?,?,?,?,?,?)",
                row,
            )
            self._conn.commit()

    # -- public API ---------------------------------------------------------
    def wrap(self, namespace: str, fn, key_fn, *, provider: str | None = None, status_fn=None):
        """Return a drop-in replacement for `fn` that reads/writes the cache.

        `fn` is a backend closure like backends.build_search()'s search(query) or
        build_scrape()'s scrape(url, mode="direct"); `key_fn` turns its args into
        the normalized cache key. Scrape is 2-arg, so use:
            search: key_fn=norm_query
            scrape: key_fn=norm_scrape, status_fn=scrape_status   # mode is in the key
        The wrapper stores the RAW, UNCAPPED response -- MAX_TOOL_CHARS stays in
        tools.py at read time. `status_fn` classifies the response for negative
        caching (defaults to ok/empty; scrape_status also flags failure sentinels
        as 'error').
        """
        if namespace not in CANNED:
            raise ValueError(f"unknown namespace {namespace!r}; add a CANNED constant for it")
        classify = status_fn or _default_status

        def wrapped(*args, **kwargs):
            key = key_fn(*args, **kwargs)
            row = self._get(namespace, key)
            if row is not None:
                # "canned" (frozen) serves ANY recorded row verbatim -- including a
                # recorded failure -- so replay is deterministic. "live"/"error"
                # serve only good rows and treat a stored 'error' as a miss, so a
                # transient failure gets re-fetched on the next populate pass.
                if self.miss_policy == "canned" or row["status"] in ("ok", "empty"):
                    self._hits += 1
                    return row["response"]

            # miss: absent key, or a stored 'error' under live/error policy.
            self._misses += 1
            if self.miss_policy == "canned":
                return CANNED[namespace]
            if self.miss_policy == "error":
                raise CacheMiss(f"{namespace} miss for key={key!r} (policy=error)")

            # live: call through, classify, store, return.
            response = fn(*args, **kwargs)
            status = classify(response)
            self._set(namespace, key, {"args": args, "kwargs": kwargs}, response, provider, status)
            self._writes += 1
            return response

        return wrapped

    def stats(self) -> dict:
        """Hit/miss/write counters for this process's lifetime."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "writes": self._writes,
            "miss_policy": self.miss_policy,
            "cache_version": self.cache_version,
        }

    def close(self):
        self._conn.close()
