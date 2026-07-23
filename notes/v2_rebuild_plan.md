# v2 rebuild plan — corpus, storage, and script restructure

Status: **BUILT** (was: spec — no code yet). The data layer (`corpus.sqlite`), the
`scripts/<stage>/` restructure, the vLLM-teacher corpus rebuild, and the per-family SFT
export described here are implemented and in use — the current v2 corpus holds 2233
teacher traces (2074 kept after review), all in the `sft` split. The DAgger columns
(§9.3) exist but student-led collection is still forward-looking. Kept as the design
reference.
Supersedes the ad-hoc v1 `data/` layout (loose `restaurants.jsonl` + `splits.json` +
`traces/` + `data/review/*` files). v1 stays on the bucket under `v1/`; v2 is a clean
rebuild under a new `v2/` prefix, not a migration.

## 1. Why a rebuild (not a migrate)

v1's data flow has three structural problems this rebuild fixes:

1. **Schema knowledge smeared across ~8 scripts.** Each script re-parses traces,
   reject-lists, and splits by hand (`build_grpo` literally imports helpers from
   `build_sft`). There is no single access layer.
2. **The split is advisory, not enforced.** `build_sft` never filters by split
   (eval-leak risk); `build_grpo` expects a split format the file doesn't use.
   The split is also only 2-way (train/eval), so SFT and GRPO can't be separated.
3. **Derived artifacts stored as if they were source.** The 16 GB `merged/`
   checkpoint (re-derivable from base+adapter) and the built datasets
   (re-derivable from traces) sit on S3 with drift/verification gaps.

The rebuild also folds in a corpus regeneration (new harvest API, self-hosted
vLLM teacher) — see §9.

## 2. Settled decisions

- **`corpus.sqlite`** is the single source of truth (one file): restaurants +
  traces + meta, merging what were `restaurants.jsonl`, `splits.json`,
  `traces/`, `reject_list.txt`, and `grounding.json`.
- **`cache.sqlite` stays a separate file** (content-hash keyed, hot during GRPO
  rollout, may grow to GBs after the large warm — different lifecycle).
- **3-way split**: `sft | grpo | eval`, disjoint by `restaurant_id`, **nullable**
  (`NULL` = unmarked). Assignment is **random-seeded**, decoupled from harvest.
- **GRPO is trace-free**: `grpo`-split restaurants get NO teacher traces; GRPO
  rows are built from `restaurants` rows + dietary sampling (reward is
  teacher-free). Traces exist only for `sft`+`eval`.
- **Built datasets are EXPORTS, not tables**: md5'd frozen `.jsonl` + a
  `.meta.json` provenance sidecar. SFT export is **per-student-family** (chat
  template is baked in); GRPO export is student-agnostic (messages, template
  applied at train time).
- **Models**: family-keyed layout; store **base + adapters + meta only**.
  Regenerate `merged/` and `merged-text/` on-pod. `merged` storage policy:
  **current-best is regenerated, never stored.**
- **Teacher**: self-hosted vLLM — now `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8`
  (4×H100, TP=4) — replacing the Claude API.
- **Analysis**: `scripts/analysis/` CLIs (not a notebook — no plots today).
- **Restaurant fields**: lean — `restaurant_id, name, city, source, is_chain,
  split`. Dropped from v1: `lat, lng, region, country, price_tier, cuisine`.
- **Out of scope this pass**: the multi-student abstraction
  (`src/students/<family>/`). Layout is forward-compat (family-keyed), but only
  `gemma-4-e4b-it` is populated. Tool-call wire format is family-specific
  (Gemma's special-token form vs. JSON/Hermes forms), so the render/parse
  refactor is deferred deliberately.

## 3. `corpus.sqlite` schema

SQLite has no native JSON type; JSON columns are `TEXT` holding JSON. All string
comparisons for grounding use the shared normalizer in `src/grounding.py`.

```sql
-- Provenance / reproducibility for the whole DB.
CREATE TABLE corpus_meta (
  key   TEXT PRIMARY KEY,
  value TEXT
);
-- rows: schema_version, created_at, harvest_source, split_seed, teacher_model

CREATE TABLE restaurants (
  restaurant_id TEXT PRIMARY KEY,   -- API-agnostic hash of name+city (stable across harvests)
  name          TEXT NOT NULL,
  city          TEXT NOT NULL,      -- with name, forms the episode input "{name}, {city}"
  source        TEXT NOT NULL,      -- provenance: 'osm' | 'yelp' | 'foursquare' | ...
  is_chain      BOOLEAN,            -- nullable: some harvest APIs don't provide it
  split         TEXT,               -- NULL = unmarked
  CHECK (split IN ('sft','grpo','eval') OR split IS NULL)
);
CREATE INDEX idx_restaurants_split ON restaurants(split);

CREATE TABLE traces (
  trace_id             TEXT PRIMARY KEY,   -- '<rid>' or '<rid>__<diet-slug>'
  restaurant_id        TEXT NOT NULL REFERENCES restaurants(restaurant_id),
  dietary_restrictions TEXT,               -- JSON: null | ["vegetarian", ...]
  -- provenance
  model                TEXT NOT NULL,      -- teacher id that produced the TARGET
  trace_source         TEXT NOT NULL DEFAULT 'teacher',  -- 'teacher' | 'student_dagger'
  dagger_round         INTEGER,            -- NULL unless trace_source='student_dagger'
  prompt_variant       TEXT NOT NULL,      -- 'teacher' (capture-time prompt)
  -- outcome
  found                BOOLEAN NOT NULL,
  schema_valid         BOOLEAN NOT NULL,
  grounding            REAL,               -- fraction in [0,1]; NULL if no menu
  unmatched_items      TEXT,               -- JSON: capped list of ungrounded item names
  -- payload
  final_json           TEXT NOT NULL,      -- JSON: the cleaned menu object
  messages             TEXT NOT NULL,      -- JSON: full trajectory (teacher content blocks)
  queries              TEXT,               -- JSON: search queries issued
  urls                 TEXT,               -- JSON: scraped URLs
  -- review
  rejected             BOOLEAN NOT NULL DEFAULT 0,   -- was reject_list.txt
  reject_reason        TEXT,
  captured_at          TEXT NOT NULL
);
CREATE INDEX idx_traces_rid    ON traces(restaurant_id);
CREATE INDEX idx_traces_source ON traces(trace_source);
```

Notes:
- `grounding` + `unmatched_items` are computed **at capture time** in
  `build_corpus` (was the separate `audit_grounding.py` post-scan). The sibling
  map that script also produced becomes a query: `siblings(rid) = SELECT trace_id
  FROM traces WHERE restaurant_id=? AND trace_id!=?`.
- `trace_source`/`dagger_round` are the DAgger forward-compat columns (§9.3).
  Default `'teacher'` means today's behavior-cloning corpus is unaffected.
- `rejected` is written by the review UI (§7), the one stateful writer besides
  `build_corpus`.

## 4. S3 `v2/` object layout

```
v2/
  corpus.sqlite                         # §3 — restaurants + traces + meta (ONE object; was 1000 trace files)
  cache.sqlite                          # tool cache; large after the GRPO warm (§9.2)

  sft/
    gemma-4-e4b-it/
      train.jsonl                       # student-rendered SFT export (md5'd)
      train.jsonl.meta.json             # provenance: corpus md5, student prompt ver, MAX_TOOL_CHARS, seed, git sha
  grpo/
    train.jsonl                         # student-agnostic prompts (md5'd)
    train.jsonl.meta.json

  models/
    gemma-4-e4b-it/
      base/                             # pinned untrained weights, md5-pinned
        <safetensors, config, tokenizer, chat_template>
        kv-backfill/                    # family-specific serving quirk (54 dead KV tensors)
      sft/<run-id>/                     # e.g. 20260714-qlora-r16
        adapter/                        # LoRA + tokenizer (~140 MB) — the ONLY stored weights
        meta.json                       # §6
      grpo/<run-id>/
        adapter/
        meta.json
      manifest.json                     # current-best sft + grpo run ids, with eval scores

  eval/<date>/<run>/
    report.json                         # scores; references the checkpoint run-id + its md5
    candidates.tgz
    <run>.log
```

**NOT stored** (regenerated on-pod): `merged/`, `merged-text/`. Regeneration =
pull `base/` + `adapter/`, merge, then `to_text_only.py --base base/kv-backfill`.
This satisfies the "serve merged bf16, never 4-bit+adapter" fidelity finding
because merge is from the **bf16** base.

**Retired entirely**: `restaurants.jsonl`, `splits.json`, `reject_list.txt`,
`data/review/grounding.json`, `data/review/decisions.json` (→ DB), and
`labels.jsonl` (a v1 phantom referenced by `cache_sync`/`eval_menu` but never
built — "findable" becomes a derived field, not a file).

## 5. Dataset export contracts

Both are `SELECT` over `corpus.sqlite` + the existing render logic, exported to a
frozen md5'd `.jsonl` with a `.meta.json` sidecar (git sha, corpus md5, seed,
prompt version, caps). Regenerable at will; never hand-edited.

- **SFT** (`build_sft`): `traces WHERE split='sft' AND NOT rejected`, re-rendered
  under the **student** prompt + the family's chat template (lossless round-trip
  verified). Per-family output. Mixes `trace_source` in (teacher + DAgger) once
  §9.3 lands; today it's all `teacher`.
- **GRPO** (`build_grpo`): `restaurants WHERE split='grpo'` + seeded dietary
  sampling → prompt message lists (`[system(student), user(episode_input)]`). No
  traces read. Student-agnostic.

## 6. Model `meta.json` (per adapter run)

The lineage record that makes a 140 MB adapter fully reproducible:

```json
{
  "family": "gemma-4-e4b-it",
  "stage": "sft",                       // or "grpo"
  "run_id": "20260714-qlora-r16",
  "base_ref": {"path": "v2/models/gemma-4-e4b-it/base", "md5": "<base safetensors md5>"},
  "starting_checkpoint": null,          // grpo: the sft run-id it initialized from
  "dataset": {"path": "v2/sft/gemma-4-e4b-it/train.jsonl", "md5": "<md5>"},
  "quant": {"method": "qlora", "quant_type": "nf4", "compute_dtype": "bfloat16"},
  "hyperparams": {"lora_r": 16, "lora_alpha": 32, "lr": 2e-4, "epochs": 3},
  "git_sha": "<sha>",
  "created_at": "<iso8601>",
  "eval": {"run": "v2/eval/.../report.json", "success": null}
}
```

`quant` is load-bearing: the v1 finding is that serving fidelity must match the
training-time base dtype, so the recipe travels with the adapter (covers the
Q-LoRA / quant experiments in §9.5). `manifest.json` per family names the
current-best sft + grpo run so "what's shipped" is never guesswork.

## 7. Script restructure

New shared layer in `src/` (the backbone — kills the smeared-schema problem):

| module | role |
|---|---|
| `src/corpus.py` | the ONLY code that knows the schema: `open_corpus()`, `iter_restaurants(split=…)`, `iter_traces(split=…, include_rejected=…)`, `write_trace(…)`, `assign_splits(seed, …)`, `set_grounding(…)`, `set_rejected(…)`, `siblings(rid)` |
| `src/grounding.py` | the single grounding normalizer+scorer (was duplicated in `audit_grounding` and `viz/review`) |
| `src/eval_metrics.py` | extend; "findable" is a derived field, retiring `labels.jsonl` |

Scripts grouped by pipeline stage (all thin CLIs over `src/corpus.py`):

```
scripts/corpus/
  harvest.py         # was harvest_restaurants.py; NEW API (§9.1); split-marking removed
                     #   → --assign-split none|random (default none)
  assign_splits.py   # NEW; random-seeded; fills NULLs freely, REFUSES to move an
                     #   already-assigned rid that has traces unless --force (leak guard)
  warm_cache.py      # --splits sft,grpo,eval ; breadth args for the big GRPO warm (§9.2)
  build_corpus.py    # vLLM teacher (§9.4); computes grounding at write; sets trace_source
scripts/datasets/
  build_sft.py       # traces WHERE split='sft' AND NOT rejected; per-family export
  build_grpo.py      # restaurants WHERE split='grpo'; trace-free
scripts/train/
  train_sft.py  train_grpo.py  to_text_only.py   # quant recipe → meta.json
scripts/eval/
  eval.py            # merges eval_split (produce) + eval_menu (score); findable field
scripts/analysis/
  analyze_queries.py  analyze_tool_chars.py       # read-only over corpus.sqlite
scripts/infra/
  corpus_sync.py     # was cache_sync.py; syncs BOTH sqlite DBs + sft/grpo/models/eval
                     #   prefixes; drops labels.jsonl; reuses the VACUUM INTO WAL snapshot
  runpod_create.py
```

**Deleted**: `audit_grounding.py` (computation→field, flagging/siblings→queries),
`eval_menu.py` (→ merged into `scripts/eval/eval.py`).

**`viz/review.py` retargeted to the DB**: reads the `grounding` field (was
`grounding.json`), writes `traces.rejected` via `corpus.set_rejected()` (was
`reject_list.txt` + `decisions.json`). It becomes a `corpus.sqlite` client — the
one interactive stateful writer. It should write to a working copy that
`corpus_sync push` then uploads.

## 8. v2 data flow (producer → consumer)

```
harvest.py ─────────────► restaurants (split=NULL)
assign_splits.py ───────► restaurants.split  (random-seeded)
warm_cache.py ──────────► cache.sqlite        (--splits; big for grpo)
build_corpus.py ────────► traces (+ grounding field)   [sft+eval restaurants; vLLM teacher]
viz/review.py ──────────► traces.rejected
build_sft.py ───────────► sft/<family>/train.jsonl     [split='sft', not rejected]
build_grpo.py ──────────► grpo/train.jsonl             [split='grpo', trace-free]
train_sft.py ───────────► models/<family>/sft/<run>/adapter + meta
train_grpo.py ──────────► models/<family>/grpo/<run>/adapter + meta   [starts from an sft run]
eval.py ────────────────► eval/<date>/<run>/report.json
corpus_sync.py ─────────► S3 v2/  (both DBs snapshotted via VACUUM INTO; md5 skip-unchanged)
analysis/* ─────────────► read-only reports (re-run after a rebuild to re-tune warm + MAX_TOOL_CHARS)
```

Operational win: syncing traces goes from ~1000 per-file objects to **one**
`corpus.sqlite` object (fewer HEAD/PUT calls — addresses the "sync serialized
every scrape" pain). Trade-off: a single new trace re-uploads the ~75 MB DB;
cheap with same-region egress + versioning.

## 9. Experiments this round & how the plan accommodates them

1. **New harvest API.** `source` column captures provenance; `is_chain`
   nullable (not every API tags chains); `restaurant_id` stays an
   API-agnostic name+city hash so re-harvests don't collide. No schema change.
2. **Much larger GRPO cache warm.** `warm_cache` gains breadth (multiple query
   templates, deeper URL funnel, both modes) so the student's *exploration*
   distribution — not just the teacher's path — is pre-cached. Goal: run GRPO
   fully **`canned`** (offline, deterministic) so the GPU never blocks on a live
   scrape. `cache.sqlite` may grow to GBs; that's the intended storage-for-GPU
   trade. No corpus schema change (cache is a separate file).
3. **DAgger (student-led + teacher-led mix).** The only schema-touching item:
   `traces.trace_source` (`teacher` | `student_dagger`) + `dagger_round`.
   Collection mode (run student on `sft`-split restaurants, relabel the visited
   states with the teacher, append as `student_dagger` traces) is a **future**
   `build_corpus --dagger` mode; the columns exist now so the schema is stable.
   `build_sft` already selects `split='sft'` and will simply include both
   sources (optionally weighted/curriculum'd later).
4. **vLLM teacher instead of Claude API.** `build_corpus` points at the vLLM
   teacher runner (`src/serving/openai_agent.py` chat+tools path);
   `traces.model` and `corpus_meta.teacher_model` record the new id. The teacher
   must be a model vLLM has a tool-call parser for.
5. **Q-LoRA / Turbo-Quant.** Training-time quant only. Each adapter's `meta.json`
   records the `quant` recipe so serve-time fidelity matches train-time base
   dtype (v1 finding). No corpus/layout change. (If "Turbo-Quant" is a specific
   library rather than a generic quant experiment, revisit — nothing here
   assumes a particular one.)

## 10. Open items before/while building

- **Split reassignment guard** in `assign_splits.py`: fill-NULL is free; moving an
  assigned rid that already has traces is a leak — refuse unless `--force`.
- **Review app write path**: `viz/review.py` writing into `corpus.sqlite`
  concurrently with a possible build — settle on a working-copy + sync flow.
- **`corpus.sqlite` size**: `messages` blobs (~73 MB in v1) live in-file per the
  one-file decision; revisit only if sync friction appears.
- **eval ↔ model lineage**: `report.json` must carry the scored checkpoint's
  `run_id` + md5 so eval results are traceable to weights.
```
