# Phase 2 — Tool-call caching & training-corpus construction

Phase 1 gave us a working agentic loop (`restaurant name -> menu JSON`) for both
Gemma ([src/gemma/agent.py](../src/gemma/agent.py)) and the Claude baseline
([src/claude/claude_agent.py](../src/claude/claude_agent.py)), backed by live
Brave (search) + local headless Chromium (scrape) tools ([src/backends.py](../src/backends.py)).

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
[.env.example](../.env.example).

| # | What | Needed for | Notes |
|---|------|-----------|-------|
| 0.1 | **AWS S3 bucket** — **DONE** (see [S3_setup.md](S3_setup.md)) | source of truth (WS-D) | `s3://restaurant-menu-corpus` (us-west-2, private, SSE-S3, BucketOwnerEnforced). Private bucket = not redistributing scraped menus. |
| 0.2 | **AWS credentials** — **DONE** (instance profile) | S3 sync | Devbox role `menu-corpus-devbox-role` grants ListBucket + Get/PutObject on the bucket; boto3 default chain resolves it. **No static keys in `.env`** — only `S3_BUCKET=restaurant-menu-corpus`, `S3_PREFIX=v1`, `AWS_DEFAULT_REGION=us-west-2` (bucket region; avoids cross-region redirects). |
| 0.3 | **Brave Search API key** | live search | Already wired as `BRAVE_API_KEY`. **Confirmed: paid tier** — quota covers the corpus build (a few searches × a few thousand restaurants); still throttle politely in WS-C. |
| 0.4 | **Local scrape (headless Chromium)** — *no key* | live scrape | scrape_url runs a local pooled Chromium (`playwright install chromium` + system libs); no API key. This is still the slow call — it's rate-limited by your own box (CPU + the single egress IP), so size corpus-build parallelism against that (and watch for per-site rate-limits on the shared IP). |
| 0.5 | **Anthropic key** | teacher SFT traces + Opus findability | Already `ANTHROPIC_API_KEY`. Budget note: findability (WS-F) uses **Opus** with a generous tool budget — estimate cost before running at full scale. |
| 0.6 | **Google Places (New) API key** — *optional* | metadata enrichment (WS-B) | Only if you want price tier / chain signals beyond OSM. Requires a **GCP project with billing enabled**; create + **restrict** the key; add `GOOGLE_PLACES_API_KEY`. Skippable — OSM alone is enough to start. |
| 0.7 | **OSM / Overpass** | bulk restaurant list | **No account, no key.** Just be polite with rate limits (or point at a specific Overpass mirror). |
| 0.8 | **Add Python deps** | several | `uv add boto3` (S3), `uv add jsonschema` (validate against `MENU_SCHEMA`). Overpass/Places use the existing `requests`. |
| 0.9 | **Scope knobs — DECIDED** | WS-B / WS-C | **English-only**; default regions = English-speaking metros (US/CA/UK/AU — WS-B ships the concrete bbox list as its CLI default); target **~3k train + ~500 eval**. WS-C3 corpus is **mixed**: 60% restriction-free + 40% dietary-conditioned (3:2 via `--conditioned-frac`), one build → one SFT run; the restriction is a visible per-episode input recorded in the trace, not distilled. These stay CLI args. |

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
scripts/build_corpus.py         # WS-C1/C3 (pilot + sized teacher run)
scripts/cache_sync.py           # WS-D
scripts/analyze_queries.py      # WS-E
scripts/label_findability.py    # WS-F
scripts/eval_menu.py            # WS-G
scripts/warm_cache.py           # WS-C2 (bulk warm, all restaurants; subsumes old WS-H)
scripts/build_sft.py            # WS-I (traces/ -> student-rendered SFT dataset)
data/sft/train.jsonl            # WS-I output: capped, student-prompt SFT examples
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

- **Key normalization** (`key_fn`): search → `norm_query` (`query.strip().lower()`,
  collapsed whitespace); scrape → **`norm_scrape` = canonical URL + `mode`**. The
  local backend's `scrape(url, mode="direct")` is **two-arg**, and `"direct"` vs
  `"browser"` return genuinely different content (browser auto-scrolls; direct's
  escalation is no-scroll) — the model's chosen mode is baked into the trajectory,
  so the two modes are **distinct cache entries**. Key on the *requested* mode;
  store whatever that mode returned (incl. a direct→browser escalation).
- **Negative caching:** store `status ∈ {ok, empty, error}` (scrape uses
  `status_fn=scrape_status`, which flags the backend's failure sentinels —
  `"(scrape failed …)"`, `"(page returned no content)"` — as `error` so a
  transient local-Chromium timeout / rate-limit isn't frozen into the corpus).
  `live`/`error` re-fetch (or raise on) `error` rows and serve only `ok`/`empty`;
  `canned` (frozen) serves **any** recorded row verbatim for deterministic replay
  and returns the canned constant only for a genuinely absent key.
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
  provider      TEXT,               -- 'brave' | 'local' | ...
  status        TEXT NOT NULL,      -- 'ok' | 'empty' | 'error'
  cache_version INTEGER NOT NULL,
  captured_at   TEXT NOT NULL       -- ISO timestamp (stamped inside Cache._set)
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

- **`restaurant_id` stability:** format lat/lng at fixed 5-decimal precision
  (`f"{lat:.5f}"`) before hashing, so the id survives re-harvests and float
  round-trips across sources.
- **`is_chain` without Places:** use OSM's `brand`/`brand:wikidata` tag when
  present; else a name-frequency heuristic across the harvest (same normalized
  name at ≥3 distinct locations → chain). Otherwise the stratification axis we
  care most about is undefined on the default (no-Places) path.

### 1.5 SFT trace (`traces/<restaurant_id>.json`)

```json
{
  "restaurant_id": "...", "restaurant_name": "...",
  "model": "claude-sonnet-5",
  "prompt_variant": "teacher",
  "dietary_restrictions": null,
  "cache_version": 1,
  "messages": [ /* full message list as sent/received */ ],
  "queries": ["search query strings, in order"],
  "urls": ["scraped urls, in order"],
  "final_json": { /* parsed answer */ } ,
  "schema_valid": true,
  "captured_at": "ISO"
}
```
`queries`/`urls` are the extracted tool-call args — the input WS-E mines.
`prompt_variant`/`dietary_restrictions`/`cache_version` record what the teacher
actually ran with: context distillation re-renders these traces under the
**student** prompt later, which is only sound if we know the teacher-side config
(`dietary_restrictions` is null for free episodes, the restriction phrase list
for conditioned ones — see WS-C3's mixed corpus).

---

## Part 2 — Workstreams (parallelizable)

Each is a self-contained agent task. Header line = **dependencies**. "Done when"
is the acceptance check.

### WS-A · Cache module `src/cache.py` + wire into `setup_tools`
**Deps:** contracts (1.2–1.3). **Independent otherwise.**

- Implement `Cache` per 1.2/1.3.
- Wire into [src/tools.py](../src/tools.py) `setup_tools(offline, cache=None)`:
  cache **wraps the backend closures** (`build_search()`/`build_scrape()` from
  [src/backends.py](../src/backends.py)) *before* `build_model_tools` applies the
  `MAX_TOOL_CHARS` cap — so the raw response is stored and the cap stays tunable.
  ```python
  search_fn, scrape_fn = build_search(), build_scrape()
  if cache:
      search_fn = cache.wrap("search", search_fn, key_fn=norm_query, provider="brave")
      scrape_fn = cache.wrap("scrape", scrape_fn, key_fn=norm_scrape,
                             status_fn=scrape_status, provider="local")
  tools, registry = build_model_tools(search_fn, scrape_fn)
  # NB: scrape is 2-arg (url, mode); norm_scrape keys on BOTH so direct/browser
  # renders are distinct entries. scrape_status marks failure sentinels 'error'.
  ```
- Add a `--cache-policy {live,canned,error,off}` flag to
  [src/gemma/run_agent.py](../src/gemma/run_agent.py) and
  [src/claude/run_claude.py](../src/claude/run_claude.py); build the `Cache` and
  pass it in. `off` = today's behavior (no cache).
- **Done when:** unit tests with a counting fake `fn` (no network) prove
  hit/miss/write behavior under all three policies, including the error-row
  re-fetch and canned-verbatim-replay rules. As a smoke check, running the same
  restaurant twice with `--cache-policy live` shows **≈0 misses** on the second
  run via `cache.stats()` (not exactly 0 — model sampling isn't deterministic,
  so the second trajectory can issue slightly different queries), and
  `--cache-policy canned` on an empty DB returns canned constants with a logged
  miss count.

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

### WS-C · Corpus build / SFT trace capture — **three stages** (restructured 2026-07-03)
**Deps:** WS-A (Cache API + `setup_tools(cache=)`), WS-B (`restaurants.jsonl`).
Reuses the existing Claude loop; `run_episode` returns `(final_text, messages)`
for trace capture (landed with Wave 0 — assistant turns hold SDK content blocks,
so serialize via `block.model_dump()` when writing trace JSON).

Restructured so the Sonnet spend is **sized after a pilot** instead of committed
up front: the Anthropic cost scales with the number of *traces* (a warm cache
doesn't make an episode cheaper), but warming decouples all the slow/fragile
scrape work from the paid calls and defers the bulk-spend decision.

**WS-C1 · Pilot** (`scripts/build_corpus.py --limit ~100`) — **~100 stratified
train restaurants**, `Cache(miss_policy="live")`, episode input `"{name},
{city}"` (matches `TEST_RESTAURANT`'s shape). Records traces (1.5); episodes are
**restriction-free**. Output feeds WS-E and the go/no-go on trace quality
(schema-valid rate, tool-call distribution, source selection) before any bulk
spend.

**WS-C2 · Bulk programmatic warm** (`scripts/warm_cache.py`, subsumes old WS-H)
— for **ALL 3500 restaurants (train + eval)**, no Anthropic tokens: run the
WS-E-mined query templates through the cached search (`live` policy), take the
top-K result URLs (funnel-domain-weighted), and scrape each.

- **Conditional mode escalation (revised 2026-07-03):** scrape each URL in
  `direct` mode; escalate the SAME URL to `browser` mode **only when the direct
  result comes back thin** (empty / clearly missing a menu — the same
  shell-detection signal the agent uses, and the same rule the teacher prompt's
  scrape strategy encodes). Do **not** blindly warm both modes for every URL:
  that ~doubles the slow browser renders (measured ~9.5 s/restaurant → ~9–10 h
  for 3500 at 2 workers, vs ~5–6 h escalating conditionally) for paths the
  student rarely takes. Warming the mode the agent *actually* requests is what
  matters for the frozen-cache hit rate; a URL that scrapes cleanly in `direct`
  is never requested in `browser`, so its `browser` entry is dead weight.
  (Rationale: on this 15 GB / no-swap / 4-core box the browser render is the
  wall-clock bottleneck, not the network — so cutting redundant renders, not
  adding bandwidth or IPs, is the lever. Keep `--workers` ≤ 3–4.)

Cheap (paid Brave + local CPU); re-runnable (`live` = no-op on warm keys,
self-heals `error` rows). This is also what makes the frozen eval + GRPO cache
possible.

**WS-C3 · Sized teacher run** (same `build_corpus.py`, higher `--limit`) — trace
count decided AFTER inspecting the pilot (all 3000? fewer? findability-
filtered?). Runs fast against the warm cache; misses still fetch live so
trajectories aren't constrained.

**WS-C3 is a MIXED corpus (free + dietary-conditioned), one build for one SFT
run.** `--conditioned-frac` sets the conditioned share of `--limit`; the sized
run uses **`--limit 1000 --conditioned-frac 0.4`** → **600 free + 400
conditioned** (a 3:2 free:conditioned split). Rationale:
- Filtering is a *narrower* skill than the full agentic loop (search behavior is
  restriction-independent; only the final "keep complying items" step differs),
  so it needs solid-minority representation, not a second full corpus. Keeping
  free dominant (60%) preserves "no restriction → full menu" as the default
  behavior and guards against over-filtering.
- A dietary restriction is a per-episode INPUT that changes the target, so it is
  **visible to both teacher and student** (recorded in `dietary_restrictions`,
  slotted into the system prompt) — it is NOT distilled away. The student learns
  the *skill* from restriction-conditioned examples, not the prompt trick.
- Conditioned episodes **reuse the front of the seeded restaurant order**
  (`rows[i % len]`, rotating `DIETARY_POOL`), so (a) every one is a warm-cache
  hit — cost is Anthropic tokens only (~$80 for 400), no new Brave/scrape — and
  (b) each pairs contrastively with that restaurant's free trace (same menu,
  restriction flips the target — exactly the conditioning signal). Trace files:
  free `<rid>.json`, conditioned `<rid>__<slug>.json` (no collision).
- `DIETARY_POOL` spreads across axes (vegetarian/vegan/gluten-free/dairy-free,
  single allergens, halal/kosher/pescatarian/keto, + a couple combos) so the
  student generalizes to unseen phrasings. Some restaurant+restriction combos
  yield `found=true, menu=[]` (nothing complies) — a valid target, kept.
- Split discipline for the later conditioned EVAL: draw it from **eval-split**
  restaurants (unseen restaurant + ideally unseen restriction phrasing) to test
  that filtering generalizes and the student (no teacher guidance) still avoids
  leaking drinks / delivery-app sources.

Shared mechanics (all stages):
- Extract `queries`/`urls` from the message list; validate `final_json` against
  `MENU_SCHEMA` (jsonschema) and set `schema_valid`.
- Parallelize with a **thread** pool, not processes: the pooled Chromium is
  thread-local (one ~200–300 MB browser per worker), so on the 15 GB no-swap box
  cap at **~3–4 workers**; threads also share one `Cache` connection. Two
  workers missing the same key both fetch and both write — benign
  (`INSERT OR REPLACE`, last-write-wins); don't "fix" it with a lock around the
  network call. Respect the per-site + Brave rate limits from Part 0.
  Idempotent: skip restaurants whose trace already exists (resumable across
  interrupted runs). Selection order is seeded and prefix-stable, so a smaller
  `--limit` run is always a subset of a larger one.
- **Done when (C1):** ~100 pilot traces exist with a printed quality summary.
  **(C2):** a `canned` replay over eval-split templates+URLs reports a low miss
  rate (print it; investigate >~10%). **(C3):** every selected restaurant has a
  trace, cache writes ≈ unique queries+urls, summary prints schema-valid rate +
  mean tool calls.

### WS-D · S3 sync `scripts/cache_sync.py`
**Deps:** contracts (1.1). Interface-only dep on WS-A (just the file path).

- `push` / `pull` for `data/cache.sqlite`, `restaurants.jsonl`, `splits.json`,
  `traces/`, `labels.jsonl` to/from `s3://$S3_BUCKET/$S3_PREFIX/...` via boto3.
- Use instance-profile creds if present, else `.env` keys. Content-hash or
  size+mtime guard to avoid needless re-uploads. `--dry-run`.
- **`cache.sqlite` is in WAL mode** — live state spans the db + `-wal`/`-shm`
  sidecars, so never upload the bare file mid-run (stale or torn snapshot).
  `push` snapshots first: `VACUUM INTO` a temp file (or
  `PRAGMA wal_checkpoint(TRUNCATE)` while no writers are active) and uploads
  the snapshot.
- **Done when:** `push` then a fresh `pull` into an empty `data/` reproduces the
  files byte-for-byte; works with an instance profile and with static keys.

### WS-E · Query analysis `scripts/analyze_queries.py`
**Deps:** WS-C1 (pilot traces) — runs on the pilot first (its templates drive
the WS-C2 warm), then re-runs over the full corpus once WS-C3 lands.

- Aggregate every `(restaurant_id, query)` and `(restaurant_id, url)` across
  `traces/`. Cluster/normalize queries into **templates** (e.g. `"{name} {city}
  menu"`, `"{name} menu"`, site-scoped like `doordash.com {name}`), report
  frequencies and per-restaurant query counts.
- Report the **URL funnel**: which domains the teacher converges on (own-site vs
  DoorDash/Yelp/etc.) — this tells WS-F/GRPO which URLs are worth pre-fetching.
- Output a short markdown report + a `query_templates.json` used to warm the
  cache deterministically for eval/held-out restaurants (consumed by WS-H).
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
*Buildable* early, but frozen runs need **WS-C2's warmed cache** — against an
unwarmed cache every tool call returns the canned constants and the metrics
measure nothing but abstention.

- Given a runner (Gemma or Claude) and the **eval split**, run episodes with
  `--cache-policy canned` (frozen) and score: schema-validity, item/section
  counts, price coverage, and **correct-abstention** (did it return empty/"not
  findable" when `labels.jsonl` says the menu isn't on the web?).
- This is the scaffold the **GRPO reward** will reuse in Phase 3 — keep the
  scoring functions importable, not buried in `__main__`.
- **Done when:** it prints a metrics table for the Claude baseline on the eval
  split (against the WS-C2-warmed cache), and abstention scoring reads
  `labels.jsonl`.

### WS-H · *(folded into WS-C2, 2026-07-03)*
The eval-split warm pass grew into the **bulk warm over all 3500 restaurants**
— see WS-C2. Rationale unchanged: a frozen (`canned`) eval against an unwarmed
cache scores nothing but abstention, and live-policy eval isn't reproducible
across days. Generalizing it to train+eval also pre-pays the GRPO frozen-cache
mitigation Part 4 deferred.

### WS-I · Trace → SFT dataset `scripts/build_sft.py` (bridge to Phase 3)
**Deps:** WS-C3 (the traces to transform). The one seam where the teacher/student
prompt swap happens; also where the SFT recipe decisions (Part 4) are implemented.

The per-file `traces/<id>.json` are the **immutable capture** (raw, model-agnostic,
the unit manual not-found filtering operates on) — *not* the training format.
`build_sft.py` is the transform that materializes the trainable dataset, and it
must do three things the raw traces deliberately don't:

- **Re-render under the *student* prompt + Gemma's chat template.** The stored
  `messages` are Anthropic content blocks generated under the *teacher* prompt.
  The SFT target is that same trajectory re-rendered with the **student** system
  prompt (`build_system_prompt(variant="student")`) through
  `tokenizer.apply_chat_template`. This is the whole point of context distillation
  and the only place the swap occurs. Mind the Gemma template's prefix-preservation
  + reasoning-guard rules (CLAUDE.md): mid-episode `reasoning` fields are stripped
  at render, so the trained tokens must match how they render at inference —
  re-render, don't hand-assemble. Also translate Anthropic tool_use/tool_result
  blocks into the Gemma bundled-turn shape the loop already uses.
- **Apply the same `MAX_TOOL_CHARS` cap the student sees at inference.** Traces
  store raw uncapped tool results (median ~79 KB, max ~815 KB of scraped
  markdown); training on uncapped results the student never sees is a
  train/inference mismatch. Cap at transform time to match `tools.py`.
- **Consolidate** the loose per-episode files into a single streamable
  `data/sft/train.jsonl` (one example/line; TRL `SFTTrainer` / HF `datasets` read
  it natively). If the repeated markdown makes it unwieldy at 1000+ traces,
  Parquet is the columnar/compressed fallback — but default to JSONL.

Filtering/labels: drop traces flagged by manual not-found review; honor the SFT
recipe decisions in Part 4 (found=false inclusion + ratio cap; reasoning
treatment). Record provenance (`restaurant_id`, `model`, `cache_version`) per
example so a dataset row traces back to its capture.
- **Done when:** `train.jsonl` re-renders every kept trace losslessly (round-trip:
  the rendered assistant turns tokenize back to the same tool calls / final JSON
  the trace recorded), the **student** prompt appears (teacher guidance absent),
  tool results are capped, and a printed summary shows found=true/false counts and
  the found=false ratio.

---

## Part 3 — Dependency graph & suggested waves

```
Wave 0 (DONE):     contracts + `data/` git-ignore + run_episode trace-return change
Wave 1 (DONE):     WS-A (cache)      WS-B (sourcing)      WS-D (s3 sync)
Wave 2 (DONE):     WS-C1 (pilot ~100 traces)          WS-G (eval harness, build-only)
Wave 3 (DONE):     WS-E on pilot traces (templates + URL funnel)
Wave 4:            WS-C2 (bulk warm, ALL 3500, direct + browser-on-thin)
Wave 5 (parallel): WS-C3 (sized teacher run, ~1000)    WS-F (findability, warm cache)
then:              WS-G frozen eval RUN; WS-E re-run over the full corpus
Wave 6:            WS-I (traces -> student-rendered SFT dataset) — bridge to Phase 3
```

- WS-C1 needs only Wave-1 output → start immediately; WS-G build alongside.
- WS-E's pilot templates drive WS-C2; WS-C2's warm cache makes WS-C3 fast and
  WS-F cheaper (Opus reads cached pages instead of waiting on scrapes).
- The WS-C3 trace count is a **decision gate** after the pilot, not a default.

To avoid file collisions when fanning out agents: each workstream owns **its own
new file(s)**; the only shared edits are WS-A touching
[src/tools.py](../src/tools.py) + the two `run_*.py` CLIs, and Wave 0 touching
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
- **English-only** sourcing (US/CA/UK/AU metros to start); WS-C3 is a **mixed
  corpus** — 60% restriction-free + 40% dietary-conditioned (3:2), built in one
  pass via `--conditioned-frac` so it feeds a single SFT run. The restriction is
  a visible per-episode input (recorded in `dietary_restrictions`), not distilled.
- Frozen (`canned`) eval over the eval split requires the **explicit warm pass
  (WS-C2)**; live-policy eval is off the table (not reproducible across days).
- **WS-C is staged (pilot → warm → sized run)**: cache warming is decoupled from
  paid trace generation; the bulk Sonnet spend is sized only after the ~100-trace
  pilot is inspected.

**Deferred to Phase 3 (GRPO) — not built now, but design leaves room:**
- **Student-explores-differently miss rate:** the frozen cache is populated from
  the *teacher's* URLs **and modes**; the student may request another URL — or the
  same URL in the other scrape `mode` — → canned miss. Mitigation **pulled into
  Phase 2 as WS-C2** (bulk warm, all restaurants; `direct` for every kept URL plus
  a `browser` entry wherever `direct` came back thin — the conditional escalation
  above, so the warmed modes match the ones the agent requests). What remains
  deferred is *measuring* the student's per-URL/per-mode miss rate before RL and
  expanding the snapshot if high — including back-filling `browser` entries for
  URLs the student requests it on but `direct` had sufficed for the teacher.
- **Reward correctness ground truth:** schema-validity + heuristics get us far,
  but true menu correctness needs the hand-labeled gold eval subset (start it in
  WS-G). This caps how well distillation can be distinguished from hallucination.
- **Cache versioning discipline:** bump `cache_version` whenever the stored
  response shape changes; never mutate rows in place.

---

## Part 5 — SFT recipe (WS-I): what goes into `train.jsonl`

The transform is mechanical; the *content* decisions below are what make or break
distillation. Some are locked, one is deliberately staged.

**Locked:**
- **Train on the `found=false` traces (kept, not dropped).** They are the
  anti-hallucination and abstention signal — a small model with no not-found
  examples learns to always emit *a* menu, the exact failure GRPO would then have
  to unlearn. They also carry the identity/exhaustion reasoning we most want
  (see below). *Guard:* keep the found=false **ratio modest and honest** (mirror
  the WS-F reward-hacking guard) — too many not-founds teaches giving up. The
  ratio is a printed WS-I summary stat; the manual not-found review removes the
  *false* negatives (tool-failure, not genuine) so what remains is real
  abstention. Rough target: hold found=false near its natural rate (~10–15% from
  the pilot), not inflated.
- **Student prompt at train time = student prompt at inference.** WS-I renders
  with `variant="student"`; never leak teacher guidance into the target.

**Open — distill the teacher's reasoning, or train action-only? (decide at WS-I):**
This is a genuine fork; here is the decision-relevant context.

- **The Gemma-template constraint is decisive.** Its reasoning-guard strips
  `reasoning`/`reasoning_content` from every assistant turn *before the last user
  message* (CLAUDE.md, "Prefix-preservation"). In a multi-tool episode the
  valuable reasoning is the **per-step** decision (is this the right restaurant?
  own-site vs delivery app? do I have enough to answer?) — all mid-episode, so
  putting the teacher's thinking into the `reasoning` field is **silently dropped
  at render**. Naive "carry Sonnet's thinking blocks across" therefore trains on
  final-turn reasoning only (which is just "format the JSON" — low value).
- **The teacher's raw CoT may not even be available.** Sonnet runs adaptive
  thinking; with summarized/omitted display the trace holds a summary or empty
  thinking, not the verbatim chain — so "distill the CoT" can mean distilling a
  summary in Sonnet's voice, not the real reasoning.
- **Three viable recipes, increasing ambition:**
  1. **Action-only (behavioral cloning).** Student emits tool calls + final JSON,
     no reasoning. Cheapest, most robust for a 4B model, and it frees the
     inference token budget (reasoning competes with `MAX_TOKENS` and the tool
     budget). The policy (source selection, identity check, persistence — now all
     in `_TEACHER_GUIDANCE`) is distilled as *action patterns*. Risk: less
     interpretable; hard behaviors like identity verification may need more
     contrastive examples since there's no verbalized check.
  2. **Final-turn reasoning only.** Low value (final turn is just formatting);
     skip.
  3. **Per-step reasoning as *visible text*, not the reasoning field.** In the
     agent loop, an assistant turn may carry visible text *before* its tool call;
     visible content is NOT stripped by the reasoning-guard and stays
     prefix-preserving. So terse, student-voice rationale ("first result is a
     DoorDash link — checking the own site instead"; "this page is a different
     city — not the same restaurant") emitted before each tool call *survives
     render* and trains the exact decisions we care about. Cost: a transform that
     condenses teacher thinking into SHORT visible lines (Sonnet's raw thinking is
     too long for a 4B), more inference tokens, and the output-format rule must
     stay scoped to the *final* answer only.
- **Recommendation (staged, matches the project's "add data, don't leak prompt"
  principle):** start with **recipe 1 (action-only)** for the first SFT run — it
  is the cheapest baseline and the WS-G eval already measures whether identity /
  source-selection behavior distilled from actions alone. **If a specific behavior
  fails to distill, add recipe-3 visible reasoning *surgically for that decision*
  (and especially on the found=false subset, where the identity/exhaustion
  rationale is the whole point)** rather than blanket-distilling CoT everywhere.
  Note this means the *capture* need not change — recipe 3 is reconstructable from
  the teacher's thinking already in the traces, so the choice stays open at WS-I
  time and does not gate the WS-C3 corpus run.

---

*Note: [CLAUDE.md](../CLAUDE.md) is current (Brave + local Chromium scrape, and its
caching note now points at this plan + [src/cache.py](../src/cache.py)).*
