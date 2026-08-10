"""Content-addressed SQLite cache for the tool-call network seam (Phase 2).

This is the WAVE-0 CONTRACT for Phase 2 (see notes/phase2_plan.md). It fixes the public
API — `Cache`, `norm_query`, `norm_url`, `CANNED`, `CacheMiss`, `MISS_POLICIES` —
so the parallel workstreams can build against stable signatures. WS-A owns
hardening (error-row caching, concurrency stress, unit tests) and wiring it into
`setup_tools`; the skeleton here is deliberately minimal but functional so
dependent workstreams can import and exercise it.

Design (locked in notes/phase2_plan.md):
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


# A bot-walled page usually answers 200 with a token body ("Access Denied", an empty
# shell) rather than an error: too little to be a menu, but non-empty, so it used to
# classify 'ok' and count as coverage. TripAdvisor returns FIFTEEN characters this
# way, and an 'ok' row is a permanent hit under live -- so four of a restaurant's
# top-3 URL slots could sit filled with nothing, invisibly. Below this many
# characters a response is 'empty' instead. Deliberately NOT 'error': 'error'
# re-fetches on every populate pass, and a site that stonewalls us will stonewall us
# again -- this is a permanent negative result, not a transient one. The floor sits
# well under the smallest real page observed (636 chars) and well over the bot-wall
# bodies (15 chars). backends.py's dead-end sentinels (BLOCKED_SITE_RESULT etc.)
# RELY on this floor: they stay under it (and match no failure marker) precisely so
# they classify 'empty' and cache as permanent negatives.
MIN_CONTENT_CHARS = 200


def scrape_status(response: str) -> str:
    """Status for a scrape response: 'error' for the backend's failure sentinels,
    'empty' for nothing (or too little to be content), else 'ok'. Pass as
    `status_fn=scrape_status` when wrapping scrape."""
    r = (response or "").strip()
    if not r:
        return "empty"
    if r.startswith(_SCRAPE_FAILURE_MARKERS):
        return "error"
    if len(r) < MIN_CONTENT_CHARS:
        return "empty"
    return "ok"


# ---------------------------------------------------------------------------
# Stored-response bound (general storage rule)
# ---------------------------------------------------------------------------
# A cache row never persists more than this many characters of response. The
# read-time MAX_TOOL_CHARS cap (tools.py) already bounds what the AGENT sees; this
# bounds what the cache STORES, so one pathological page cannot bloat the shared
# cache.sqlite (a single 14M-char PDF scrape helped push the pilot cache to 2.15
# GB). Kept well ABOVE MAX_TOOL_CHARS so the stored row still contains everything
# any read-time cap would surface -- lowering the read cap stays retunable without
# re-scraping. _set enforces it on every new write; clip_oversized() retro-applies
# it to a cache built before the rule (then vacuum() reclaims the freed pages).
MAX_STORED_CHARS = 400_000


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
        # check_same_thread=False lets the WS-C thread pool share this one
        # connection; the _lock below -- NOT WAL -- is what makes that safe:
        # every read/write is serialized in Python, so SQLite never sees
        # concurrent access and WAL's own reader/writer concurrency goes unused.
        # WAL + synchronous=NORMAL is kept purely for cheap commits (append to
        # -wal, fsync deferred to checkpoint).
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        # self._lock above only serializes THREADS inside one process. Multi-PROCESS
        # writers (2-GPU GRPO runs one trainer rank per GPU, each with its own Cache on
        # the same file) are serialized by SQLite itself: WAL permits exactly one writer,
        # and the default busy_timeout of 0 makes the loser raise
        # `OperationalError: database is locked` IMMEDIATELY rather than wait. Under live
        # rollouts both ranks write a scrape row every few seconds, so without this a
        # multi-GPU run dies on a lock collision. Writes here are single small rows, so
        # any real contention clears in milliseconds; 30 s is a ceiling, not a target.
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # -- key + row helpers --------------------------------------------------
    def _key_hash(self, namespace: str, key: str) -> str:
        raw = f"{namespace}\x00{key}\x00{self.cache_version}".encode()
        return hashlib.sha256(raw).hexdigest()

    def _get(self, namespace: str, key: str) -> sqlite3.Row | None:
        # The lock serializes ALL access to the shared connection. Reads need it
        # too: concurrent execute() calls on one sqlite3 connection clobber each
        # other's in-flight cursors -- observed as InterfaceError("bad parameter
        # or other API misuse") AND, worse, one thread receiving another thread's
        # row (silent cross-talk) under the WS-C thread pool.
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM cache WHERE key_hash = ?", (self._key_hash(namespace, key),)
            )
            return cur.fetchone()

    def _set(self, namespace, key, args, response, provider, status):
        # General storage rule (see MAX_STORED_CHARS): never persist a monster
        # response. `status` was already classified on the FULL response by the
        # caller, so a huge page is still 'ok'; only the STORED text is bounded.
        if response is not None and len(response) > MAX_STORED_CHARS:
            response = response[:MAX_STORED_CHARS]
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
    def wrap(self, namespace: str, fn, key_fn, *, provider: str | None = None, status_fn=None,
             store_if=None):
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

        `store_if(response) -> bool` (optional) vetoes STORAGE while still returning
        the response to the caller. Use it for responses that are not answers from
        the network at all -- pass store_if=backends.is_cacheable on scrape so a
        broken local browser cannot write rows for URLs it never fetched.
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
                    with self._lock:  # += is a non-atomic read-modify-write
                        self._hits += 1
                    return row["response"]

            # miss: absent key, or a stored 'error' under live/error policy.
            with self._lock:
                self._misses += 1
            if self.miss_policy == "canned":
                return CANNED[namespace]
            if self.miss_policy == "error":
                raise CacheMiss(f"{namespace} miss for key={key!r} (policy=error)")

            # live: call through, classify, store, return. The network call runs
            # OUTSIDE the lock (two workers missing the same key both fetch and
            # both write -- benign, INSERT OR REPLACE; see notes/phase2_plan.md WS-C).
            response = fn(*args, **kwargs)
            if store_if is not None and not store_if(response):
                # Returned but NOT stored: this response says something about the
                # local machine, not about the URL (see backends.is_cacheable).
                # Writing it would record a finding that was never made, and under
                # "canned" it would later replay as if the page had answered it.
                # Leaving the key absent is self-healing -- the next pass re-fetches.
                return response
            status = classify(response)
            self._set(namespace, key, {"args": args, "kwargs": kwargs}, response, provider, status)
            with self._lock:
                self._writes += 1
            return response

        return wrapped

    def stats(self) -> dict:
        """Hit/miss/write counters for this process's lifetime."""
        with self._lock:  # consistent snapshot across the three counters
            return {
                "hits": self._hits,
                "misses": self._misses,
                "writes": self._writes,
                "miss_policy": self.miss_policy,
                "cache_version": self.cache_version,
            }

    def clip_oversized(self, max_chars: int = MAX_STORED_CHARS, *, dry_run: bool = False) -> dict:
        """Retro-apply the MAX_STORED_CHARS storage rule to rows already stored.

        Clips every response longer than `max_chars` to its first `max_chars`
        characters -- the same bound _set now enforces on write -- so a cache built
        before the rule can be cleaned in place. Returns a report
        {'rows', 'chars_removed', 'largest_before', 'max_chars', 'applied'}; a
        dry_run reports the same counts without mutating. The UPDATE only frees
        pages INSIDE the file; call vacuum() afterward to actually shrink it on disk.
        """
        with self._lock:
            n, removed, largest = self._conn.execute(
                "SELECT count(*), coalesce(sum(length(response) - ?), 0), "
                "coalesce(max(length(response)), 0) "
                "FROM cache WHERE length(response) > ?",
                (max_chars, max_chars),
            ).fetchone()
            if n and not dry_run:
                self._conn.execute(
                    "UPDATE cache SET response = substr(response, 1, ?) "
                    "WHERE length(response) > ?",
                    (max_chars, max_chars),
                )
                self._conn.commit()
        return {
            "rows": int(n), "chars_removed": int(removed), "largest_before": int(largest),
            "max_chars": max_chars, "applied": bool(n and not dry_run),
        }

    def slim_rows(self, slim_fn, namespace: str = "scrape", *, dry_run: bool = False) -> dict:
        """Bake a read-time slimming transform into every stored response in a namespace.

        `slim_fn(response) -> str` is the SAME transform the backend applies (pass
        backends._slim_scrape): it drops base64 data: URIs, markdown images, empty
        links/bullets, and dead hrefs, and collapses blank runs. Storing the RAW
        response keeps the MAX_TOOL_CHARS *cap* retunable, but the slim removes pure
        noise the read path already strips on every hit -- so persisting it just makes
        the file match what the agent sees (minus the cap) and reclaims the base64
        bulk that dominates a cache captured before source-side scrubbing (a single
        inline image measured 397K chars; it can also be clipped mid-string so the
        read-time image regex can't catch it). Only rows whose slimmed text differs
        are rewritten, so it is idempotent -- a second run changes nothing.

        Scrape-only by default: search results are not slimmed on read (the model
        mines them for URLs), so slimming them here would drift storage from the read
        path. Returns {'namespace','rows_scanned','rows_changed','chars_removed',
        'largest_before','applied'}; dry_run reports the same without mutating. Frees
        pages inside the file only -- call vacuum() afterward to shrink it on disk.
        """
        scanned = changed = removed = largest = 0
        with self._lock:
            # Collect keys first, then fetch+update one row at a time: bounds peak
            # memory to a single response (some are ~400K) instead of loading the
            # whole namespace, and avoids mutating a live SELECT cursor mid-iteration.
            key_hashes = [
                r["key_hash"] for r in self._conn.execute(
                    "SELECT key_hash FROM cache WHERE namespace = ?", (namespace,)
                ).fetchall()
            ]
            for kh in key_hashes:
                resp = self._conn.execute(
                    "SELECT response FROM cache WHERE key_hash = ?", (kh,)
                ).fetchone()["response"]
                scanned += 1
                if resp is None:
                    continue
                if len(resp) > largest:
                    largest = len(resp)
                slim = slim_fn(resp)
                if slim != resp:
                    changed += 1
                    removed += len(resp) - len(slim)
                    if not dry_run:
                        self._conn.execute(
                            "UPDATE cache SET response = ? WHERE key_hash = ?", (slim, kh)
                        )
            if changed and not dry_run:
                self._conn.commit()
        return {
            "namespace": namespace, "rows_scanned": scanned, "rows_changed": changed,
            "chars_removed": removed, "largest_before": largest,
            "applied": bool(changed and not dry_run),
        }

    def reclassify(self, status_fn, namespace: str = "scrape", *, dry_run: bool = False) -> dict:
        """Recompute `status` from the STORED response for every row in a namespace.

        slim_rows/clip_oversized rewrite `response` only, so a row classified on its
        PRE-slim text keeps that verdict forever -- e.g. a bot-wall page whose raw
        markdown was 300 chars of image junk classified 'ok' when stored, but slims
        to under MIN_CONTENT_CHARS and should be 'empty' (a permanent negative).
        Under "live" that stale 'ok' is a permanent hit serving junk. Run this AFTER
        slim_rows with the same status_fn the read path uses (cache.scrape_status)
        so storage and classification agree.

        Returns {'namespace', 'rows_scanned', 'rows_changed', 'transitions',
        'applied'}; `transitions` counts each change as "old->new". dry_run reports
        without mutating. Rows with a NULL response are left untouched.
        """
        scanned = changed = 0
        transitions: dict[str, int] = {}
        with self._lock:
            rows = self._conn.execute(
                "SELECT key_hash, status, response FROM cache WHERE namespace = ?",
                (namespace,),
            ).fetchall()
            for row in rows:
                scanned += 1
                if row["response"] is None:
                    continue
                status = status_fn(row["response"])
                if status != row["status"]:
                    changed += 1
                    key = f"{row['status']}->{status}"
                    transitions[key] = transitions.get(key, 0) + 1
                    if not dry_run:
                        self._conn.execute(
                            "UPDATE cache SET status = ? WHERE key_hash = ?",
                            (status, row["key_hash"]),
                        )
            if changed and not dry_run:
                self._conn.commit()
        return {
            "namespace": namespace, "rows_scanned": scanned, "rows_changed": changed,
            "transitions": transitions, "applied": bool(changed and not dry_run),
        }

    def vacuum(self) -> None:
        """Rebuild the db file to reclaim pages freed by clip_oversized/slim_rows (VACUUM).
        Needs free disk roughly equal to the current db size for the temp copy."""
        with self._lock:
            self._conn.execute("VACUUM")

    def close(self):
        # Fold the WAL back into the main db and truncate the -wal sidecar so a
        # clean shutdown leaves a self-contained single file. (A crash can still
        # leave -wal lagging the bare db; WS-D's VACUUM INTO snapshot in
        # scripts/cache_sync.py covers that case -- always sync via that script,
        # never a manual copy of data/cache.sqlite.)
        with self._lock:
            try:
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.Error:
                pass  # busy/failed checkpoint must not turn close() into a raise
            self._conn.close()
