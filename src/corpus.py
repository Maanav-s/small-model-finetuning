"""The v2 corpus store: one SQLite DB that is the single source of truth for
restaurants, the sft/grpo/eval split, and teacher/DAgger traces.

Replaces the v1 loose files (`restaurants.jsonl` + `splits.json` + `traces/*.json`
+ `data/review/reject_list.txt` + `data/review/grounding.json`). See
notes/v2_rebuild_plan.md for the full spec; the DDL below is §3 of that plan.

Everything that touches the corpus goes through this module -- it is the ONE
place that knows the schema, so scripts stay thin CLIs (no more re-parsing traces
/ reject-lists / splits by hand in eight different files).

  from corpus import open_corpus, trace_id_for
  with open_corpus("data/corpus.sqlite") as cx:
      cx.upsert_restaurants(rows)
      cx.assign_splits(seed=42, fractions={"sft": 0.5, "grpo": 0.3, "eval": 0.2})
      for r in cx.iter_restaurants(split="sft"):
          ...

Design notes:
  * SQLite has no JSON type; JSON columns are TEXT holding `json.dumps(...)`.
    Reads parse them back, so callers always see Python objects (the same trace
    dict shape v1 wrote, plus joined restaurant name/city + a computed
    `episode_input`).
  * WAL mode + synchronous=NORMAL, matching src/cache.py, so a corpus build and
    the review UI can read/write without blocking. corpus_sync snapshots via
    `VACUUM INTO` (folds the WAL) before upload -- do not upload the bare file
    mid-write.
  * `restaurant_id` is an API-agnostic hash of name+city (see `restaurant_id_for`)
    so a re-harvest from a different source produces stable ids.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

SCHEMA_VERSION = 2

# Columns stored as JSON TEXT -- encoded on write, decoded on read.
_TRACE_JSON_COLS = (
    "dietary_restrictions",
    "unmatched_items",
    "final_json",
    "messages",
    "queries",
    "urls",
)

VALID_SPLITS = ("sft", "grpo", "eval")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS corpus_meta (
  key   TEXT PRIMARY KEY,
  value TEXT
);

CREATE TABLE IF NOT EXISTS restaurants (
  restaurant_id TEXT PRIMARY KEY,
  name          TEXT NOT NULL,
  city          TEXT NOT NULL,
  source        TEXT NOT NULL,
  is_chain      INTEGER,            -- nullable: not every harvest API tags chains
  split         TEXT,               -- NULL = unmarked
  CHECK (split IN ('sft','grpo','eval') OR split IS NULL)
);
CREATE INDEX IF NOT EXISTS idx_restaurants_split ON restaurants(split);

CREATE TABLE IF NOT EXISTS traces (
  trace_id             TEXT PRIMARY KEY,
  restaurant_id        TEXT NOT NULL REFERENCES restaurants(restaurant_id),
  dietary_restrictions TEXT,        -- JSON: null | ["vegetarian", ...]
  model                TEXT NOT NULL,
  trace_source         TEXT NOT NULL DEFAULT 'teacher',   -- 'teacher' | 'student_dagger'
  dagger_round         INTEGER,
  prompt_variant       TEXT NOT NULL,
  found                INTEGER NOT NULL,
  schema_valid         INTEGER NOT NULL,
  grounding            REAL,        -- fraction in [0,1]; NULL if no menu
  unmatched_items      TEXT,        -- JSON list of ungrounded item names
  final_json           TEXT NOT NULL,   -- JSON
  messages             TEXT NOT NULL,   -- JSON: full trajectory
  queries              TEXT,        -- JSON
  urls                 TEXT,        -- JSON
  cache_version        INTEGER,
  parse_error          TEXT,
  rejected             INTEGER NOT NULL DEFAULT 0,
  reject_reason        TEXT,
  reviewed_at          TEXT,        -- NULL = not yet human-reviewed (viz/review.py)
  captured_at          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_traces_rid    ON traces(restaurant_id);
CREATE INDEX IF NOT EXISTS idx_traces_source ON traces(trace_source);
"""


# ---------------------------------------------------------------------------
# id / trace-key derivation (shared so harvest + build_corpus never drift)
# ---------------------------------------------------------------------------
def restaurant_id_for(name: str, city: str) -> str:
    """Stable, API-agnostic restaurant id: 16 hex chars of sha1(name|city).

    Normalized (lowercase, collapsed whitespace) so trivial formatting
    differences between harvest sources don't mint duplicate ids. Matches the
    16-char id width v1 used.
    """
    norm = lambda s: re.sub(r"\s+", " ", (s or "").strip().lower())
    h = hashlib.sha1(f"{norm(name)}|{norm(city)}".encode("utf-8"))
    return h.hexdigest()[:16]


def restriction_slug(restrictions: list[str]) -> str:
    """Filesystem/id-safe tag for a normalized restriction list."""
    slug = re.sub(r"[^a-z0-9]+", "-", "-".join(restrictions).lower()).strip("-")
    return slug[:60] or "diet"


def trace_id_for(restaurant_id: str, restrictions: list[str] | None) -> str:
    """Trace key: free -> '<rid>'; conditioned -> '<rid>__<slug>' (matches the v1
    trace *filename* stem, minus the .json, so ids carry over conceptually)."""
    if not restrictions:
        return restaurant_id
    return f"{restaurant_id}__{restriction_slug(restrictions)}"


# ---------------------------------------------------------------------------
def _to_bool_int(v: Any) -> int | None:
    if v is None:
        return None
    return 1 if v else 0


def _json_dumps(v: Any) -> str | None:
    if v is None:
        return None
    return json.dumps(v, ensure_ascii=False)


def _json_loads(v: str | None) -> Any:
    if v is None:
        return None
    return json.loads(v)


class Corpus:
    """Handle on a corpus.sqlite. Prefer `open_corpus(...)` / the context manager."""

    def __init__(self, path: str | Path, *, create: bool = True):
        self.path = str(path)
        if not create and not Path(self.path).exists():
            raise FileNotFoundError(f"no corpus at {self.path}")
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._migrate()
        if self.get_meta("schema_version") is None:
            self.set_meta("schema_version", str(SCHEMA_VERSION))
        self._conn.commit()

    def _migrate(self) -> None:
        """Additive schema migrations for DBs created by an earlier build. New
        columns default to NULL, so a plain ADD COLUMN is safe + backward-compatible."""
        cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(traces)")}
        if "reviewed_at" not in cols:
            self._conn.execute("ALTER TABLE traces ADD COLUMN reviewed_at TEXT")

    # -- meta ---------------------------------------------------------------
    def set_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO corpus_meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        self._conn.commit()

    def get_meta(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM corpus_meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    # -- restaurants --------------------------------------------------------
    def upsert_restaurant(self, row: dict) -> None:
        self.upsert_restaurants([row])

    def upsert_restaurants(self, rows: Iterable[dict]) -> int:
        """Insert/update restaurants. `split` is preserved if the incoming row
        omits it (so a re-harvest doesn't wipe an assigned split); pass split
        explicitly to set it. Returns the number of rows processed."""
        n = 0
        for r in rows:
            rid = r.get("restaurant_id") or restaurant_id_for(r["name"], r["city"])
            has_split = "split" in r
            if has_split:
                self._conn.execute(
                    "INSERT INTO restaurants(restaurant_id, name, city, source, is_chain, split) "
                    "VALUES(:rid, :name, :city, :source, :is_chain, :split) "
                    "ON CONFLICT(restaurant_id) DO UPDATE SET "
                    "name=excluded.name, city=excluded.city, source=excluded.source, "
                    "is_chain=excluded.is_chain, split=excluded.split",
                    {"rid": rid, "name": r["name"], "city": r["city"],
                     "source": r.get("source", "unknown"),
                     "is_chain": _to_bool_int(r.get("is_chain")), "split": r.get("split")},
                )
            else:
                # Preserve existing split on conflict.
                self._conn.execute(
                    "INSERT INTO restaurants(restaurant_id, name, city, source, is_chain, split) "
                    "VALUES(:rid, :name, :city, :source, :is_chain, NULL) "
                    "ON CONFLICT(restaurant_id) DO UPDATE SET "
                    "name=excluded.name, city=excluded.city, source=excluded.source, "
                    "is_chain=excluded.is_chain",
                    {"rid": rid, "name": r["name"], "city": r["city"],
                     "source": r.get("source", "unknown"),
                     "is_chain": _to_bool_int(r.get("is_chain"))},
                )
            n += 1
        self._conn.commit()
        return n

    def get_restaurant(self, restaurant_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM restaurants WHERE restaurant_id=?", (restaurant_id,)
        ).fetchone()
        return _restaurant_row_to_dict(row) if row else None

    def iter_restaurants(
        self, *, split: str | None = None, unmarked: bool = False
    ) -> Iterator[dict]:
        """Yield restaurant dicts. `split` filters to one split; `unmarked=True`
        yields only rows with no split (split IS NULL). With neither, yields all.
        Ordered by restaurant_id for a stable, seedable base order."""
        if unmarked:
            cur = self._conn.execute(
                "SELECT * FROM restaurants WHERE split IS NULL ORDER BY restaurant_id"
            )
        elif split is not None:
            _check_split(split)
            cur = self._conn.execute(
                "SELECT * FROM restaurants WHERE split=? ORDER BY restaurant_id", (split,)
            )
        else:
            cur = self._conn.execute("SELECT * FROM restaurants ORDER BY restaurant_id")
        for row in cur:
            yield _restaurant_row_to_dict(row)

    def count_by_split(self) -> dict[str, int]:
        """{'sft': n, 'grpo': n, 'eval': n, 'unmarked': n}."""
        out = {s: 0 for s in VALID_SPLITS}
        out["unmarked"] = 0
        for row in self._conn.execute(
            "SELECT split, COUNT(*) AS n FROM restaurants GROUP BY split"
        ):
            out["unmarked" if row["split"] is None else row["split"]] = row["n"]
        return out

    def assign_splits(
        self,
        *,
        seed: int,
        fractions: dict[str, float],
        reassign: bool = False,
        force: bool = False,
    ) -> dict[str, int]:
        """Random-seeded split assignment.

        Default (reassign=False): assign ONLY currently-unmarked restaurants,
        leaving existing assignments untouched -- always safe (fill-from-NULL).

        reassign=True: (re)assign ALL restaurants. This can MOVE a restaurant to a
        different split; if that restaurant already has traces, moving it is a
        leakage hazard (a trace built for 'sft' resurfacing under 'eval'), so it is
        REFUSED unless force=True.

        `fractions` maps split -> share (must cover the valid splits and ~sum 1.0).
        Assignment is deterministic given (seed, the id set, fractions): sort ids,
        seeded-shuffle, slice by cumulative fraction. Returns the resulting
        per-split counts among the assigned pool.
        """
        bad = set(fractions) - set(VALID_SPLITS)
        if bad:
            raise ValueError(f"unknown split(s) in fractions: {sorted(bad)}")
        if abs(sum(fractions.values()) - 1.0) > 1e-6:
            raise ValueError(f"fractions must sum to 1.0, got {sum(fractions.values())}")

        import random

        if reassign:
            pool = [r["restaurant_id"] for r in self.iter_restaurants()]
            if not force:
                with_traces = {
                    row["restaurant_id"]
                    for row in self._conn.execute(
                        "SELECT DISTINCT restaurant_id FROM traces"
                    )
                }
                if with_traces:
                    raise ValueError(
                        f"reassign would move {len(with_traces)} restaurant(s) that already "
                        "have traces (leakage risk). Pass force=True to override."
                    )
        else:
            pool = [r["restaurant_id"] for r in self.iter_restaurants(unmarked=True)]

        pool.sort()
        random.Random(seed).shuffle(pool)

        # Slice by cumulative fraction. Any rounding remainder lands in the last split.
        order = [s for s in VALID_SPLITS if s in fractions]
        counts = {s: 0 for s in order}
        n = len(pool)
        idx = 0
        for i, s in enumerate(order):
            take = n - idx if i == len(order) - 1 else round(n * fractions[s])
            for rid in pool[idx: idx + take]:
                self._conn.execute("UPDATE restaurants SET split=? WHERE restaurant_id=?", (s, rid))
                counts[s] += 1
            idx += take
        self._conn.commit()
        self.set_meta("split_seed", str(seed))
        return counts

    # -- traces -------------------------------------------------------------
    def write_trace(self, trace: dict) -> str:
        """Upsert one trace. Accepts the same dict shape build_corpus builds; the
        `trace_id` is derived from restaurant_id + dietary_restrictions if absent.
        Returns the trace_id."""
        rid = trace["restaurant_id"]
        restrictions = trace.get("dietary_restrictions")
        tid = trace.get("trace_id") or trace_id_for(rid, restrictions)
        payload = {
            "trace_id": tid,
            "restaurant_id": rid,
            "dietary_restrictions": _json_dumps(restrictions),
            "model": trace["model"],
            "trace_source": trace.get("trace_source", "teacher"),
            "dagger_round": trace.get("dagger_round"),
            "prompt_variant": trace["prompt_variant"],
            "found": _to_bool_int(trace.get("found")),
            "schema_valid": _to_bool_int(trace.get("schema_valid")),
            "grounding": trace.get("grounding"),
            "unmatched_items": _json_dumps(trace.get("unmatched_items")),
            "final_json": _json_dumps(trace.get("final_json")),
            "messages": _json_dumps(trace.get("messages")),
            "queries": _json_dumps(trace.get("queries")),
            "urls": _json_dumps(trace.get("urls")),
            "cache_version": trace.get("cache_version"),
            "parse_error": trace.get("parse_error"),
            "rejected": _to_bool_int(trace.get("rejected", 0)) or 0,
            "reject_reason": trace.get("reject_reason"),
            "reviewed_at": trace.get("reviewed_at"),
            "captured_at": trace["captured_at"],
        }
        cols = ", ".join(payload)
        placeholders = ", ".join(f":{k}" for k in payload)
        updates = ", ".join(f"{k}=excluded.{k}" for k in payload if k != "trace_id")
        self._conn.execute(
            f"INSERT INTO traces({cols}) VALUES({placeholders}) "
            f"ON CONFLICT(trace_id) DO UPDATE SET {updates}",
            payload,
        )
        self._conn.commit()
        return tid

    def get_trace(self, trace_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT t.*, r.name AS restaurant_name, r.city AS city, r.split AS split "
            "FROM traces t JOIN restaurants r ON r.restaurant_id = t.restaurant_id "
            "WHERE t.trace_id=?",
            (trace_id,),
        ).fetchone()
        return _trace_row_to_dict(row) if row else None

    def has_trace(self, trace_id: str) -> bool:
        return self._conn.execute(
            "SELECT 1 FROM traces WHERE trace_id=?", (trace_id,)
        ).fetchone() is not None

    def iter_traces(
        self,
        *,
        split: str | None = None,
        include_rejected: bool = False,
        trace_source: str | None = None,
    ) -> Iterator[dict]:
        """Yield trace dicts (JSON columns parsed; joined restaurant_name/city +
        a computed `episode_input`). `split` filters by the restaurant's split;
        `include_rejected=False` (default) drops rejected traces; `trace_source`
        filters teacher vs student_dagger. Ordered by trace_id (stable)."""
        where, params = [], []
        if split is not None:
            _check_split(split)
            where.append("r.split=?")
            params.append(split)
        if not include_rejected:
            where.append("t.rejected=0")
        if trace_source is not None:
            where.append("t.trace_source=?")
            params.append(trace_source)
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        cur = self._conn.execute(
            "SELECT t.*, r.name AS restaurant_name, r.city AS city, r.split AS split "
            "FROM traces t JOIN restaurants r ON r.restaurant_id = t.restaurant_id "
            f"{clause} ORDER BY t.trace_id",
            params,
        )
        for row in cur:
            yield _trace_row_to_dict(row)

    def siblings(self, restaurant_id: str, *, exclude: str | None = None) -> list[str]:
        """trace_ids of the other slices (free + dietary) of the same restaurant."""
        rows = self._conn.execute(
            "SELECT trace_id FROM traces WHERE restaurant_id=? ORDER BY trace_id",
            (restaurant_id,),
        ).fetchall()
        return [r["trace_id"] for r in rows if r["trace_id"] != exclude]

    def set_grounding(
        self, trace_id: str, grounding: float | None, unmatched_items: list[str] | None
    ) -> None:
        self._conn.execute(
            "UPDATE traces SET grounding=?, unmatched_items=? WHERE trace_id=?",
            (grounding, _json_dumps(unmatched_items), trace_id),
        )
        self._conn.commit()

    def set_rejected(self, trace_id: str, rejected: bool, reason: str | None = None) -> None:
        self._conn.execute(
            "UPDATE traces SET rejected=?, reject_reason=? WHERE trace_id=?",
            (1 if rejected else 0, reason, trace_id),
        )
        self._conn.commit()

    def set_review_decision(
        self, trace_id: str, decision: str, reason: str | None = None
    ) -> None:
        """Record a human review decision (viz/review.py). `decision` is 'keep' |
        'reject' | 'undecided'. keep/reject stamp `reviewed_at` (now) and set
        `rejected`; 'undecided' clears both (un-reviews the trace)."""
        if decision not in ("keep", "reject", "undecided"):
            raise ValueError(f"decision must be keep|reject|undecided, got {decision!r}")
        if decision == "undecided":
            self._conn.execute(
                "UPDATE traces SET rejected=0, reject_reason=NULL, reviewed_at=NULL WHERE trace_id=?",
                (trace_id,),
            )
        else:
            self._conn.execute(
                "UPDATE traces SET rejected=?, reject_reason=?, reviewed_at=? WHERE trace_id=?",
                (1 if decision == "reject" else 0, reason,
                 datetime.now(timezone.utc).isoformat(), trace_id),
            )
        self._conn.commit()

    def review_counts(self) -> dict[str, int]:
        """{'reviewed', 'unreviewed', 'kept', 'rejected'} over all traces -- the
        review-progress summary the UI shows."""
        row = self._conn.execute(
            "SELECT "
            "SUM(reviewed_at IS NOT NULL) AS reviewed, "
            "SUM(reviewed_at IS NULL) AS unreviewed, "
            "SUM(reviewed_at IS NOT NULL AND rejected=0) AS kept, "
            "SUM(rejected=1) AS rejected FROM traces"
        ).fetchone()
        return {k: int(row[k] or 0) for k in ("reviewed", "unreviewed", "kept", "rejected")}

    def trace_count(self, *, include_rejected: bool = True) -> int:
        q = "SELECT COUNT(*) AS n FROM traces"
        if not include_rejected:
            q += " WHERE rejected=0"
        return self._conn.execute(q).fetchone()["n"]

    # -- lifecycle ----------------------------------------------------------
    def commit(self) -> None:
        self._conn.commit()

    def close(self) -> None:
        try:
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.Error:
            pass
        self._conn.close()

    def __enter__(self) -> "Corpus":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _check_split(split: str) -> None:
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {VALID_SPLITS}, got {split!r}")


def _restaurant_row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    if d.get("is_chain") is not None:
        d["is_chain"] = bool(d["is_chain"])
    return d


def _trace_row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    for col in _TRACE_JSON_COLS:
        if col in d:
            d[col] = _json_loads(d[col])
    for col in ("found", "schema_valid", "rejected"):
        if d.get(col) is not None:
            d[col] = bool(d[col])
    name, city = d.get("restaurant_name"), d.get("city")
    if name and city:
        d["episode_input"] = f"{name}, {city}"
    return d


@contextmanager
def open_corpus(path: str | Path, *, create: bool = True) -> Iterator[Corpus]:
    """Context-managed Corpus handle (closes + checkpoints the WAL on exit)."""
    cx = Corpus(path, create=create)
    try:
        yield cx
    finally:
        cx.close()
