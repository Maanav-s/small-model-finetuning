# Results

The **permanent, in-repo record of every scored eval run.** `data/` is git-ignored and
pods are ephemeral, so this directory is the one place a result survives without S3 or
W&B. `notes/experiments.md` is the narrative log ("what we learned"); this is the
machine-readable evidence behind it.

## Run-sets in this directory

| run-set | plan | models | what it answers |
|---|---|---|---|
| [eval500-20260808](eval500-20260808/) | `eval` / seed 42 / n=500 / cond 0.4 | Qwen3-235B teacher, Gemma SFT, Gemma base | **Did SFT work?** Yes — item F1 0.438 → 0.560 over base, false-finds 29 → 10, and the student matches the teacher's find rate. This pass also *created* the eval-split reference traces every paired score since is measured against. |
| [eval500-20260810](eval500-20260810/) | same plan | Gemma GRPO (ckpt-50), Gemma SFT | **Did GRPO beat SFT?** No — item F1 0.539 vs 0.559, 71 wins / 75 losses per episode, all \|t\| < 1.96. SFT was **re-measured in the same session on the same cache** rather than quoted from 08-08, so the two rows are a true paired A/B. Paired deltas and t-stats are in [notes/experiments.md](../notes/experiments.md). |

Both use the same plan, so per-episode trace ids join **across** run-sets as well as within
one — the SFT row appears in both and moves only by cache/serving noise (F1 0.560 vs 0.559),
which is itself a useful reproducibility check.

## Layout

```
results/
  <run-set>/                     e.g. eval500-20260807
    README.md                    generated comparison table (scripts/eval/summarize.py)
    <model>.json                 one eval report per model (scripts/eval/eval.py --json)
    candidates/                  git-ignored -- archived to S3 instead (see below)
```

A **run-set** is one plan evaluated by several models: same `--split/--seed/--limit/
--conditioned-frac`, so every model runs the *same* episodes and their per-episode
trace ids join. Never mix plans inside a run-set — the table would compare models on
different restaurants.

Bulky artifacts live in S3, not here:

| artifact | home |
|---|---|
| report JSONs + the table | **this directory** (committed) |
| per-episode candidate traces | `s3://$S3_BUCKET/$S3_PREFIX/eval/<run-set>/candidates/` |
| the reference traces themselves | `corpus.sqlite` (`eval` split), synced to S3 |
| live curves, per-episode telemetry | W&B project `menu-eval` |

## Reading a report

Each `<model>.json` carries the scores **and** the conditions that produced them:

- `mode` — `paired` (scored against the teacher's reference traces: item precision/
  recall/F1, found-accuracy, abstention buckets) or `self-report` (reference-free:
  schema-valid, found=true, sizes, price coverage). The teacher is necessarily
  `self-report` — it *is* the reference, so pairing it against itself would print
  1.000 and mean nothing.
- `aggregate.{all,free,conditioned}` — every metric, sliced by whether the episode had
  a dietary restriction. The free/conditioned split is load-bearing: v1 first read
  conditioned episodes as a quality problem and they turned out to be a *termination*
  problem.
- `checkpoint` — `{run_id, md5, meta}` from the scored checkpoint's `meta.json`, so a
  number traces back to exact weights.
- `cache` — `{hits, misses, writes, hit_rate, miss_policy}`. **Read this before
  trusting a score.** A low hit rate means the model explored off the warmed
  distribution, so its result partly measures the cache, not the model; under a
  `canned` policy those lookups return a constant and the episode scores as an
  abstention.
- `run` — throughput, worker count, and the full list of failed episodes.

## Regenerating the table

```bash
uv run python scripts/eval/summarize.py results/<run-set> -o results/<run-set>/README.md
```

## Producing a run-set

`scripts/infra/eval_pod.sh` drives the whole thing on one pod, phase by phase (see the
header of that file). The ordering constraint that matters: **the teacher runs first**,
because its pass is what writes the eval-split reference traces the students are then
scored against.
