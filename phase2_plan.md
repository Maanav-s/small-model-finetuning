# Phase 2 — Tool-call caching & training-corpus construction

Phase 1 gave us a working agentic loop (`restaurant name -> menu JSON`) for both
Gemma ([src/gemma/agent.py](src/gemma/agent.py)) and the Claude baseline
([src/claude/claude_agent.py](src/claude/claude_agent.py)), backed by live
Brave (search) + Jina (scrape) tools ([src/backends.py](src/backends.py)).

Phase 2 turns that into something we can *train* on:

1. A **content-addressed SQLite cache** wrapping the two network calls, with a
   pluggable **miss policy** (`live | canned | error`) so the same code serves
   SFT (live-fallback), GRPO (frozen), and the eventual product (live-fallback).
2. A **stratified restaurant corpus** (~2–5k restaurants) sourced free from
   OSM/Overpass, optionally enriched with Google Places.
3. A **corpus-build pass** that runs the Claude teacher over the restaurants,
   records SFT traces, and populates the cache as a side effect.
4. Supporting pieces: **S3 sync** (source of truth), **query analysis**,
   **findability labeling**, and a small **eval/validation** harness.

The cache is really a **frozen fixture dataset with a capture date**, not a live
cache — that framing drives most decisions below (immutable snapshot, versioned
artifact, deterministic misses for RL).

---

## Part 0 — Manual setup YOU must do (agents can't)

Do these first; the workstreams assume the keys/resources exist. Put all secrets
in the repo-root `.env` (git-ignored) and mirror the names into
[.env.example](.env.example).

| # | What | Needed for | Notes |
|---|------|-----------|-------|
| 0.1 | **AWS S3 bucket**, private, *Block Public Access = ON* | source of truth (WS-D) | e.g. `s3://<you>-menu-corpus`. Confirms the privacy requirement: private bucket = not redistributing scraped menus. |
| 0.2 | **AWS credentials** for the training node | S3 sync | Prefer an **EC2 instance profile / IAM role** on the training box (no static keys, free same-region egress). Otherwise an IAM user with `s3:GetObject/PutObject/ListBucket` on that bucket, keys in `.env` as `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`. Add `S3_BUCKET` + `S3_PREFIX` to `.env`. |
| 0.3 | **Brave Search API key** | live search | Already wired as `BRAVE_API_KEY`. Confirm the plan's rate/quota tier is enough for the corpus build (a few searches × a few thousand restaurants). |
| 0.4 | **Jina key** (recommended, technically optional) | live scrape | `r.jina.ai` works keyless at low rate; a key (`JINA_API_KEY`) raises limits and matters for a parallel corpus build. This is the slow/expensive call — size your rate limit here. |
| 0.5 | **Anthropic key** | teacher SFT traces + Opus findability | Already `ANTHROPIC_API_KEY`. Budget note: findability (WS-F) uses **Opus** with a generous tool budget — estimate cost before running at full scale. |
| 0.6 | **Google Places (New) API key** — *optional* | metadata enrichment (WS-B) | Only if you want price tier / chain signals beyond OSM. Requires a **GCP project with billing enabled**; create + **restrict** the key; add `GOOGLE_PLACES_API_KEY`. Skippable — OSM alone is enough to start. |
| 0.7 | **OSM / Overpass** | bulk restaurant list | **No account, no key.** Just be polite with rate limits (or point at a specific Overpass mirror). |
| 0.8 | **Add Python deps** | several | `uv add boto3` (S3), `uv add jsonschema` (validate against `MENU_SCHEMA`). Overpass/Places use the existing `requests`. |
| 0.9 | **Decide scope knobs** | WS-B / WS-C | English-only? Which countries/cities? Target restaurant count (recommend ~3k train + ~500 eval to start). These become CLI args, but pick defaults now. |

You do **not** need a HuggingFace account for this phase (S3 is the artifact
store). HF login is still only required to pull the gated Gemma weights (Phase 1).

---

## Part 1 — Shared contracts (pin these BEFORE fanning out)

These interfaces are the seams between workstreams. Freeze them first (one short
commit) so agents build against stable contracts and don't collide. Everything
below is designed so **WS-A and WS-B start immediately and in parallel**, and the
rest layer on once these files exist.

### 1.1 File / directory layout

```
data/                      # NEW, git-ignored (add `data/` to .gitignore)
  cache.sqlite             # the content-addressed cache (WS-A)
  restaurants.jsonl        # the sourced corpus rows (WS-B)
  splits.json              # {restaurant_id: "train"|"eval"} (WS-B)
  traces/<restaurant_id>.json   # per-episode SFT traces (WS-C)
  labels.jsonl             # findability + menu_source_type labels (WS-F)
src/cache.py               # NEW cache module (WS-A)
scripts/harvest_restaurants.py  # WS-B
scripts/build_corpus.py         # WS-C
scripts/cache_sync.py           # WS-D
scripts/analyze_queries.py      # WS-E
scripts/label_findability.py    # WS-F
scripts/eval_menu.py            # WS-G
```

`data/` is git-ignored; its **source of truth is S3** (WS-D syncs it).

### 1.2 Cache module API (`src/cache.py`) — the central contract

```python
class Cache:
    def __init__(self, path: str, *, miss_policy: str = "live",
                 cache_version: int = 1): ...
        # miss_policy: "live"  -> on miss, call fn, store result, return it
        #              "canned"-> on miss, DO NOT call fn; return a namespace
        #                         canned constant; count+log the miss (frozen/GRPO)
        #              "error" -> on miss, raise CacheMiss (strict debugging)

    def wrap(self, namespace: str, fn, key_fn) -> callable:
        """Return a drop-in replacement for `fn` that reads/writes the cache.
        namespace: "search" | "scrape"
        key_fn:    (*args, **kwargs) -> str   normalized cache key
        Stores the RAW, UNCAPPED response (the MAX_TOOL_CHARS cap stays in
        tools.py at read time, so it can be retuned without re-scraping)."""

    def stats(self) -> dict:  # {hits, misses, writes, by_namespace...}
```

- **Key normalization** (`key_fn`): search → `query.strip().lower()` with
  collapsed whitespace; scrape → canonical URL (drop fragment + tracking params,
  normalize trailing slash, lowercase host).
- **Negative caching:** store `status ∈ {ok, empty, error}`. `live` re-fetches
  `error` rows but serves `ok`/`empty`. `canned` serves any stored row and only
  returns the canned constant for a genuine absence.
- **Canned constants** (deterministic, per namespace): search →
  `"(no results)"`, scrape → `"(page not available)"`. Keep them stable — they
  become part of the frozen training distribution.
- **SQLite pragmas:** `journal_mode=WAL` (concurrent readers during parallel
  builds/rollouts), `synchronous=NORMAL`, optional `mmap_size` / place on tmpfs
  for the fully-in-RAM read path on the big training node.

### 1.3 SQLite schema

```sql
CREATE TABLE IF NOT EXISTS cache (
  key_hash      TEXT PRIMARY KEY,   -- sha256(namespace | key | cache_version)
  namespace     TEXT NOT NULL,      -- 'search' | 'scrape'
  key           TEXT NOT NULL,      -- normalized query or canonical url
  args_json     TEXT NOT NULL,      -- original args, for debugging
  response      TEXT,               -- RAW uncapped provider response
  provider      TEXT,               -- 'brave' | 'jina' | ...
  status        TEXT NOT NULL,      -- 'ok' | 'empty' | 'error'
  cache_version INTEGER NOT NULL,
  captured_at   TEXT NOT NULL       -- ISO date (passed in; wall-clock at call site)
);
CREATE INDEX IF NOT EXISTS idx_ns ON cache(namespace);
```

### 1.4 Restaurant row (`restaurants.jsonl`, one JSON object per line)

```json
{
  "restaurant_id": "sha1(name|lat|lng)[:16]",
  "name": "string", "city": "string", "region": "string", "country": "string",
  "lat": 0.0, "lng": 0.0,
  "cuisine": ["string"], "price_tier": 0,
  "is_chain": false,
  "source": "osm" | "places" | "yelp"
}
```
`menu_source_type` and `findable` are **not** set here — WS-F fills them into
`labels.jsonl` keyed by `restaurant_id` (kept separate so labeling can re-run
without touching the source list).

### 1.5 SFT trace (`traces/<restaurant_id>.json`)

```json
{
  "restaurant_id": "...", "restaurant_name": "...",
  "model": "claude-sonnet-4-6",
  "messages": [ /* full message list as sent/received */ ],
  "queries": ["search query strings, in order"],
  "urls": ["scraped urls, in order"],
  "final_json": { /* parsed answer */ } ,
  "schema_valid": true,
  "captured_at": "ISO"
}
```
`queries`/`urls` are the extracted tool-call args — the input WS-E mines.

---

## Part 2 — Workstreams (parallelizable)

Each is a self-contained agent task. Header line = **dependencies**. "Done when"
is the acceptance check.

### WS-A · Cache module `src/cache.py` + wire into `setup_tools`
**Deps:** contracts (1.2–1.3). **Independent otherwise.**

- Implement `Cache` per 1.2/1.3.
- Wire into [src/tools.py](src/tools.py) `setup_tools(offline, cache=None)`:
  cache **wraps the backend closures** (`build_search()`/`build_scrape()` from
  [src/backends.py](src/backends.py)) *before* `build_model_tools` applies the
  `MAX_TOOL_CHARS` cap — so the raw response is stored and the cap stays tunable.
  ```python
  search_fn, scrape_fn = build_search(), build_scrape()
  if cache:
      search_fn = cache.wrap("search", search_fn, key_fn=norm_query)
      scrape_fn = cache.wrap("scrape", scrape_fn, key_fn=norm_url)
  tools, registry = build_model_tools(search_fn, scrape_fn)
  ```
- Add a `--cache-policy {live,canned,error,off}` flag to
  [src/gemma/run_agent.py](src/gemma/run_agent.py) and
  [src/claude/run_claude.py](src/claude/run_claude.py); build the `Cache` and
  pass it in. `off` = today's behavior (no cache).
- **Done when:** running the same restaurant twice with `--cache-policy live`
  makes zero network calls on the second run (assert via `cache.stats()`), and
  `--cache-policy canned` on an empty DB returns canned constants with a logged
  miss count. Unit test with a fake `fn` (no network).

### WS-B · Restaurant sourcing `scripts/harvest_restaurants.py`
**Deps:** contracts (1.4). **Fully independent (no cache, no model).**

- Query **Overpass** (`amenity=restaurant`, with `name`, `cuisine`, `addr:*`)
  across a configurable list of regions/bounding boxes; page results politely.
- Optional `--enrich-places` pass to add `price_tier` / `is_chain` signal via
  Google Places (guard behind `GOOGLE_PLACES_API_KEY`; skip cleanly if absent).
- Dedup by `restaurant_id`; drop closed/non-restaurant rows.
- **Stratified sampling** to a target count across the axes we care about:
  geography, cuisine, and **chain vs independent** (deliberately oversample
  independents — that's where tool use matters and menus are hard to find).
  Emit a small **distribution report** (counts per axis) to stdout so you can
  eyeball balance.
- Write `restaurants.jsonl` + a `splits.json` train/eval split (stratified,
  disjoint).
- **Done when:** `restaurants.jsonl` has ~target rows, `splits.json` is disjoint,
  and the distribution report shows no axis is degenerate (e.g. not 95% chains).

### WS-C · Corpus build / SFT trace capture `scripts/build_corpus.py`
**Deps:** WS-A (Cache API + `setup_tools(cache=)`), WS-B (`restaurants.jsonl`).
Reuses the existing Claude loop as-is.

- For each **train** restaurant: run `claude_agent.run_episode` with a
  `Cache(miss_policy="live")` — this **populates the cache as a side effect**
  (the SFT run *is* the cache-population pass) and records the trace (1.5).
- Extract `queries`/`urls` from the message list; validate `final_json` against
  `MENU_SCHEMA` (jsonschema) and set `schema_valid`.
- Parallelize with a worker pool (IO-bound; Jina is the bottleneck) — respect
  the Jina/Brave rate limits from Part 0. Idempotent: skip restaurants whose
  trace already exists (resumable across interrupted runs).
- **Done when:** every train restaurant has a trace, the cache is populated
  (`cache.stats()` writes ≈ unique queries+urls), and a summary prints
  schema-valid rate + mean tool calls per episode.

### WS-D · S3 sync `scripts/cache_sync.py`
**Deps:** contracts (1.1). Interface-only dep on WS-A (just the file path).

- `push` / `pull` for `data/cache.sqlite`, `restaurants.jsonl`, `splits.json`,
  `traces/`, `labels.jsonl` to/from `s3://$S3_BUCKET/$S3_PREFIX/...` via boto3.
- Use instance-profile creds if present, else `.env` keys. Content-hash or
  size+mtime guard to avoid needless re-uploads. `--dry-run`.
- **Done when:** `push` then a fresh `pull` into an empty `data/` reproduces the
  files byte-for-byte; works with an instance profile and with static keys.

### WS-E · Query analysis `scripts/analyze_queries.py`
**Deps:** WS-C (traces exist).

- Aggregate every `(restaurant_id, query)` and `(restaurant_id, url)` across
  `traces/`. Cluster/normalize queries into **templates** (e.g. `"{name} {city}
  menu"`, `"{name} menu"`, site-scoped like `doordash.com {name}`), report
  frequencies and per-restaurant query counts.
- Report the **URL funnel**: which domains the teacher converges on (own-site vs
  DoorDash/Yelp/etc.) — this tells WS-F/GRPO which URLs are worth pre-fetching.
- Output a short markdown report + a `query_templates.json` we can use to warm
  the cache deterministically for eval/held-out restaurants.
- **Done when:** the report exists and query templates cover the bulk of
  observed teacher searches; note explicitly what fraction is *not* covered.

### WS-F · Findability + menu-source labeling `scripts/label_findability.py`
**Deps:** WS-A, WS-B. Runs after (or alongside) WS-C.

- For each restaurant, determine **`findable`** at *build time* with a generous
  budget (more queries/URLs than the runtime loop; Opus as the judge). Menu
  found → `findable=true` + ensure it's cached; budget exhausted → `findable=
  false (as of <date>)`. Also record `menu_source_type ∈ {own_site, pdf,
  aggregator, image_only, social_only, none}`.
- Write `labels.jsonl` keyed by `restaurant_id`. **Guard against reward-hacking:**
  keep the `findable=false` fraction honest and modest; flag it in the summary.
- **Done when:** every restaurant has a label; a **hand-check sample** of the
  `findable=false` calls is dumped for you to spot-verify (these are the
  highest-risk labels — false negatives teach the model to give up).

### WS-G · Eval / validation harness `scripts/eval_menu.py`
**Deps:** contracts + WS-B splits. Uses `jsonschema` + `schema.extract_json`.

- Given a runner (Gemma or Claude) and the **eval split**, run episodes with
  `--cache-policy canned` (frozen) and score: schema-validity, item/section
  counts, price coverage, and **correct-abstention** (did it return empty/"not
  findable" when `labels.jsonl` says the menu isn't on the web?).
- This is the scaffold the **GRPO reward** will reuse in Phase 3 — keep the
  scoring functions importable, not buried in `__main__`.
- **Done when:** it prints a metrics table for the Claude baseline on the eval
  split, and abstention scoring reads `labels.jsonl`.

---

## Part 3 — Dependency graph & suggested waves

```
Wave 0 (you, 1 commit):  Part 1 contracts + `data/` in .gitignore + Part 0 deps/keys
Wave 1 (parallel):       WS-A (cache)      WS-B (sourcing)      WS-D (s3 sync)
Wave 2 (parallel):       WS-C (corpus/traces)   WS-F (findability)   WS-G (eval harness)
Wave 3:                  WS-E (query analysis — needs WS-C traces)
```

- WS-A, WS-B, WS-D depend only on the frozen contracts → launch together.
- WS-C and WS-F both need WS-A's `Cache` + WS-B's list → next wave.
- WS-G needs only contracts + splits; can go in Wave 1 or 2.
- WS-E strictly needs WS-C output → last.

To avoid file collisions when fanning out agents: each workstream owns **its own
new file(s)**; the only shared edits are WS-A touching
[src/tools.py](src/tools.py) + the two `run_*.py` CLIs, and Wave 0 touching
`.gitignore` + `.env.example`. Land Wave 0 first and WS-A's `setup_tools` change
early so Wave 2 agents import a stable signature.

---

## Part 4 — Decisions locked / deferred

**Locked this phase:**
- SQLite content-addressed cache (no Redis); rely on OS page cache / tmpfs for
  the RAM tier on the training node.
- S3 (private) as source of truth; `data/` is a synced, git-ignored artifact.
- OSM/Overpass first for sourcing; Places optional enrichment.
- One cache, three miss policies: **SFT = live**, **GRPO = canned (frozen)**,
  **product = live** — selected by a flag.
- Findability is a **build-time label**, not a runtime judgment; Opus judges with
  a generous budget; hand-spot-check the negatives.

**Deferred to Phase 3 (GRPO) — not built now, but design leaves room:**
- **Student-explores-differently miss rate:** the frozen cache is populated from
  the *teacher's* URLs; the student may request others → canned miss. WS-C should
  optionally pre-fetch the "obvious URL set" per restaurant (WS-E's funnel) to
  shrink this; measure the miss rate before RL and expand the snapshot if high.
- **Reward correctness ground truth:** schema-validity + heuristics get us far,
  but true menu correctness needs the hand-labeled gold eval subset (start it in
  WS-G). This caps how well distillation can be distinguished from hallucination.
- **Cache versioning discipline:** bump `cache_version` whenever the stored
  response shape changes; never mutate rows in place.

---

*Note: [CLAUDE.md](CLAUDE.md) is current (Brave + Jina, and its caching note now
points at this plan + [src/cache.py](src/cache.py)). The only stale-tooling
reference left is in `project_plan.md` (still mentions Tavily) — out of scope for
this build.*
