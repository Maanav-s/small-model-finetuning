# Experiment log

A running, append-only record of training/eval runs and their results. Newest entries
at the top. Each entry: what ran, the config that matters, the numbers, the takeaway, and
where the artifacts live (S3 is the source of truth — pods are ephemeral).

Conventions:
- **Eval plan** is the seed-reproducible mix from `scripts/corpus/build_corpus.py` (`load_seeded_rows` →
  `plan_episodes`): `--split eval --seed 42 --conditioned-frac 0.4`, rendered with the **student**
  prompt (what we ship). Same plan across models → per-episode filenames line up.
- **Self-report** metrics (no teacher reference): `schema-valid %`, `found=true %`, `mean items`.
- S3 bucket `restaurant-menu-corpus`, prefix `v1`, region `us-west-2`.

---

## 2026-08-10 — GRPO run 2 (50 steps, 2×B200): the policy moved, the task did not

The fixes from the 2026-08-09 post-mortem all landed and all did what they were supposed
to. The model still did not get better. **GRPO is not paying for itself yet.**

**Config:** 2×B200 180 GB DDP colocate, pod `khn3njeymdf2yi` ($13.58/hr), G=16,
`max_completion_length` 24576, bs=1 × accum 16 (global 32 = 2 prompts/step), lr **1e-5**,
temp **1.2**, `max_tool_calls` 6, `max_tool_chars` 16000, `vllm_gpu_memory_utilization`
0.14 + `vllm_max_model_length` 32768, cache `live`. 872 train prompts + a **30-prompt
held-out probe** every 10 steps at `num_generations_eval=2`. Stopped at step 55/150;
adapter = `checkpoint-50`. ~640-830 s/step. Artifacts:
`v2/models/gemma-4-e4b-it/grpo/gemma-menu-grpo-v2/`, eval under
`v2/eval/grpo-gemma-menu-grpo-v2/`, W&B `menu-grpo/wju9smmb`.

**The learning rate was the diagnosis and the fix worked.** `lora/b_norm_vs_sft` (now
logged every step through TRL's own `_metrics`) rose monotonically to **0.042** by step
50 — vs 0.0063 for the whole previous 100-step run at 1e-6, i.e. **~13× the movement in
half the steps.** Two steps at 1e-5 passed the entire earlier run. Whatever is wrong now,
"the optimizer never moved the policy" is no longer it.

**The probe (steps 10→50), which is the point of this run:**

| step | reward | struct | found | ground | clip | calls |
|---|---|---|---|---|---|---|
| 10 | 0.536 | 0.883 | 0.667 | 0.376 | 0.317 | 14.9 |
| 30 | 0.533 | 0.917 | 0.667 | 0.361 | 0.267 | 15.3 |
| 50 | 0.539 | 0.967 | 0.750 | 0.326 | 0.200 | 12.7 |

Reward is FLAT (spread 0.09 ≈ 2 SEM over 5 points). What did move is **cost**, not
quality: `clipped_ratio` 0.32→0.20 and `call_frequency` 14.9→12.7. The policy is learning
to finish inside its budget, not to build better menus.

**Paired eval, 500 episodes, same seeded plan, both models served in one session on one
cache (n=498 paired; SFT re-measured rather than quoted from 2026-08-08):**

| metric | SFT | GRPO | paired delta | t |
|---|---|---|---|---|
| schema-valid | 0.998 | 0.998 | 0.000 | — |
| found accuracy | 0.791 | 0.791 | 0.000 | — |
| **item F1** | **0.559** | **0.539** | **−0.018** | −1.77 |
| precision | 0.832 | 0.819 | −0.004 | −0.52 |
| recall | 0.576 | 0.558 | −0.016 | −1.58 |
| price agreement | 0.922 | 0.916 | −0.009 | −1.00 |

**Nothing is significant** (all |t| < 1.96) and the per-episode F1 record is **71 wins /
75 losses** — a coin flip. 50 steps of GRPO produced a policy statistically
indistinguishable from the SFT student, trending very slightly worse. `found_accuracy` is
identical to four decimals, which is itself telling: the reward's `found` term is not
changing behaviour at all.

**Takeaway / where to look next.** The reward moved the things it directly pays for
(termination, tool-call count) and not the thing we care about (menu completeness —
recall 0.558 vs the teacher's reference). Candidate explanations, cheapest first: (1) 50
steps × 2 prompts = 100 restaurants is simply too little data; (2) the grounding term
rewards faithfulness to scraped evidence, which a model can maximize by reporting FEWER
items — note `item_count_delta` got *worse*, −5.8 → −6.6, exactly the direction a
precision-flavoured reward pushes; (3) `structure_reward` saturated at ~0.97 contributes
no advantage, so two-thirds of the weight vector is inert. (2) is the one worth testing —
it predicts that GRPO as currently specified *cannot* fix the recall gap, because the
reward is not asking for recall.

### Traps paid for on this run (all fixed in the branch)

- **`to_text_only.py` silently produced a RANDOM model.** Merging a GRPO adapter yields an
  already-text-only checkpoint; the remap only handled `model.language_model.*` prefixes,
  so `new_sd` came out nearly empty and `load_state_dict(strict=False)` left **665 tensors
  at random init**. It printed `missing=665` and saved anyway. The artifact loaded and
  served fine and scored **0/500 schema-valid** while SFT scored 499/500 in the same
  session; weight norms sat ~28% below source (q_proj 63.1→45.5). Now passes unknown keys
  through and **exits non-zero on any missing tensor**.
- **The coherence probe that proved nothing.** `"The capital of France is"` → `"France is
  France is..."` looked like proof of corruption — until the *known-good* SFT model
  produced the identical loop. These students are SFT'd so hard onto one templated task
  that raw-text probes are worthless as health checks. The real evidence was the tensor
  norms.
- **`PRAGMA busy_timeout=30000` on `cache.sqlite`.** The in-process `threading.Lock`
  cannot serialize two DDP ranks; WAL allows one writer and sqlite's default timeout of 0
  raises `database is locked` on first collision. Invisible on one GPU.
- **`expandable_segments`.** The 2-GPU smoke OOM'd holding **51.3 GiB reserved but
  unallocated** — variable-length rollouts plus the probe's second length distribution.
- **vLLM util has a FLOOR.** 0.12 never reached a rollout: vLLM sizes its KV pool to serve
  one request at the model's max seq len (Gemma-4 defaults to 131072 → 2.18 GiB) and
  refuses to start. `--vllm-max-model-len 32768` makes the same budget hold ~4× more
  sequences.
- **Blackwell needs a self-consistent pip CUDA toolchain.** vLLM picks FlashInfer on
  sm_100 and JIT-compiles trtllm-gen FMHA, failing three times in sequence (~4 min apart,
  each after the 15 GB policy load): `ninja` not on PATH; nvcc 13.2 vs runtime-13.0
  headers; nvvm emitting PTX 9.2 for a 13.0 ptxas; then `-lcudart` in a `lib64/` the wheel
  lacks. Pin the nvcc set DOWN to 13.0.
- **Don't `| tee` TRL's output into a tmux pane.** Its rich completions table on an fd
  wandb's console capture left non-blocking killed a launch 11 min in with
  `BlockingIOError`.

---

## 2026-08-08 — GRPO launch debugging: TRL's parse loses Gemma finals (bug #11) + the real backward memory model

Standing up the real GRPO run (1×H200 SECURE $4.59/hr, pod `p0lbxgcnvn06vi`) surfaced one
silent showstopper and one wrong memory estimate. Both are now fixed on `main`
(`2165be0`, `547ee2d`); the epoch run launched clean afterwards.

**Bug #11 — every reward 0, gradient 0, no error anywhere.** The canned 2-step smoke and the
first live steps both logged `rewards/*/mean = 0, reward_std = 0, frac_reward_zero_std = 1,
grad_norm = 0` while `tools/call_frequency` was healthy (12–27) and TRL's completion table
visibly showed menu JSON. Dumping the actual reward inputs (a `REWARD_DEBUG_DUMP` probe in
`_resolve`, 1-step live run) showed **16/16 final assistant messages arrived with
`content=''`** — the JSON was gone before the reward ever saw it. Root cause chain:

- TRL ≥1.9 parses each generated segment into messages with `tokenizer.parse_response(ids,
  prefix=...)` (grpo_trainer.py ~2125) instead of handing the reward raw text.
- For a FRESH single turn that parse is correct (verified directly on the pod: thinking split
  out, content intact, `<turn|>` or not — the eval harness's `<turn|>` bug is NOT this).
- What breaks is the MID-EPISODE shape: Gemma bundles tool calls + tool responses INSIDE one
  assistant turn, so the final answer is a **turn continuation** whose prefix (built by token
  concatenation, ending `<tool_response|>`, no fresh `<|turn>model` anchor) plus a **second
  thought span** confuses the streaming response parser (`transformers/utils/chat_parsing`) —
  content is dropped or truncated (observed both `''` and 201/244-char mid-word cuts).
- The smoke's zeros had looked explainable as the documented canned+8192 pathology
  (sentinel-junk evidence → thrash → budget clipping), which is REAL but was masking bug #11:
  the live run at 16384 with `clipped_ratio 0.31` and 68% clean terminations still scored
  all-zero. **Lesson: "all rewards zero" has more than one sufficient cause — keep digging
  until the observed zeros are OVER-determined, not just explained once.**

**Fix (`547ee2d`):** `make_grpo_rewards(tokenizer=...)` now decodes the `completion_ids`
kwarg TRL always passes and reads the rollout off the RAW wire text itself: evidence = the
`<|tool_response>` spans, final answer = the tail after the last tool span with tool-call and
thought spans stripped (both can carry braces; `extract_json` decodes from the first `{`).
Priority stays: explicit `final_json`/`evidence` kwargs → raw ids → TRL-parsed messages.
Validated on-pod (1 live step): structure 0→**1.0**, found 0.25, grounding +0.08,
grad_norm 0→**0.054**, `clipped_ratio 0` at 16384. Five unit tests pin the wire path
(grounded / hallucinated / clipped-thought / dangling-call / no-tokenizer).

**Bug #10.5 — the fp32-logits estimate missed the retained graph (2 OOMs).** CLAUDE.md's
"bs=1 @ 24K ≈ 25 GB logits" counts ONE tensor; the loss chain retains ~3.5× that (bf16
logits ~12 GB → `.float()` 24 GB → log-softmax 24 GB, then a 24 GB grad buffer in
`backward()`). At `max_completion_length=24576` that is ~85 GB of logits-chain memory —
over 141 GB with policy (15) + vLLM colocate share, at ANY `vllm_gpu_memory_utilization`
(shrinking vLLM 0.30→0.20 just let the retained chain grow 10 GB further before the same
24 GB ask failed: 125→135 GiB in use). **Diagnostic that settled it: the OOM stack is in
`accelerator.backward()`, and "in use" GREW when vLLM shrank.** Fix: 16384 + util 0.18 →
peak ~114 GB, fits with ~25 GB headroom. 16384 is not binding so far (`clipped_ratio 0`
on validation; watch it on the epoch).

**Also fixed pre-launch (`2165be0`): the ToolFailureAbort callback watched a metric TRL
never logs** (`tools/failure_rate`; the real name is `tools/failure_frequency` on 1.5.1
through 1.9.2) — the dead-browser abort could never have fired. Same 0.8 threshold carries
over (it is failures/calls in [0,1]).

**Run config that launched:** 902 prompts (541 free + 361 conditioned @ 0.4, seed 42), G=8,
`max_completion_length` 16384, bs=1 × accum 16 (effective 16 → 2 prompts/step, ~451
steps/epoch), lr 1e-6, beta 0, temp 1.0, `--cache-policy live` over the fully-warmed grpo
split, vLLM colocate util 0.18, W&B project `menu-grpo`, checkpoints → S3
`v2/models/gemma-4-e4b-it/grpo/gemma-menu-grpo/` every 10 min. Step-time datapoints so far:
417 s (first pass, cache-warming the student's explorations) → 103 s (validation step over
the now-warm cache).

## 2026-08-08 — FIRST three-model v2 eval: teacher vs SFT student vs untrained base (n=500 each)

The first run where all three models are scored on the **same seeded plan** (`--split eval
--seed 42 --limit 500 --conditioned-frac 0.4` = 300 free + 200 conditioned), and the first
with a **paired** reference at all: `corpus.sqlite` had ZERO eval-split traces before this,
so the teacher pass *created* the reference the students are scored against. That is also
why the teacher row is self-report — it IS the reference; pairing it against itself would
print P=R=F1=1.000.

One 4×H100 pod ($13.16/hr): teacher on TP=4, then both Gemmas served concurrently (GPU0/GPU1,
merged bf16 text-only) and evaluated in parallel. 500/500 episodes for every model, **0
failures**. `--cache-policy live` throughout, against the fully-warmed eval split (601/601
restaurants × 6 query templates), so hit rates were 92–94% — near-frozen comparisons.

| model | mode | schema-valid | found acc. | item F1 | P | R | false-find | cache hit | eps/min |
|---|---|---|---|---|---|---|---|---|---|
| **Qwen3-235B teacher** | self-report | 99.6% | 82.0% *(found=true)* | — | — | — | — | — | 18.0 |
| **Gemma SFT student** | paired | **100.0%** | **81.3%** | **0.560** | 0.827 | 0.573 | **10** | 94.0% | 40.1 |
| **Gemma base (untrained)** | paired | 95.4% | 74.1% | 0.438 | 0.737 | 0.459 | 29 | 92.1% | 16.4 |

**The student has closed the FIND gap: 81.3% vs the teacher's 82.0%.** What it has not closed
is COMPLETENESS — precision 0.827 but recall 0.573, mean 7.8 fewer items than the reference.
When it answers it is right; it just returns a partial menu. That is precisely what GRPO's
completeness reward targets, and it is now measured rather than assumed.

**SFT beats base on every axis**, most sharply on calibration: false-finds (claiming a menu
for a restaurant that has none) **29 → 10**, false give-ups 100 → 83 of 410 findable.

**The base model is stronger than the v1 log implies** — 74.1% found accuracy and 95.4%
schema-validity UNTRAINED. So schema-following largely comes free with Gemma-4; SFT buys
precision, completeness, and calibration. Base is also 2.4× slower (16.4 vs 40.1 eps/min):
it takes many more tool calls to get there.

**Conditioned episodes are the remaining quality gap, and this time it is NOT termination.**
SFT free F1 0.655 vs conditioned 0.399 (recall 0.680 → 0.395), while found accuracy is FLAT
across slices (81.3% both). Contrast [[v1-sft-failure-mode]], where the conditioned deficit
was a termination artifact: the model now finds and answers conditioned episodes fine, it
just over-filters.

### Four silent bugs, all of which faked the v1 empty-output failure

Every one produced `final_json=None` — the exact signature of v1 non-termination — so each
would have been reported as "the v2 student is broken". Caught only by tracing a live
episode turn by turn and seeing the model emit a complete, valid menu that our code threw away:

1. **`parse_response` needs `prefix=`** (transformers 5.14.1). Without it every turn RAISED,
   the loop's recovery told the model its (valid) tool calls were unparseable, and the model
   burned all 8 tool calls apologising before returning ''.
2. **A `<turn|>`-terminated answer parses to `content=''`.** Measured: `'…<channel|>{json}<turn|>'`
   → `{'role','thinking'}`, the same text without the marker → content present. A 13,691-char
   turn carrying a full 40-item menu scored as an empty episode.
3. **The two checkpoints disagree about `prefix=`.** The SFT tokenizer (re-saved by a newer
   transformers at train time) REQUIRES it; the base gemma-4-E4B-it tokenizer is legacy
   `response_schema` and REJECTS it — with a ValueError, not a TypeError. Fixing (1) fixed SFT
   and broke base.
4. **`--gemma-max-tokens`**: the 4096 default truncates this student mid-answer
   (`finish_reason=length`), which also yields empty content. Real, but NOT the main cause —
   diagnosed first and fixed nothing on its own. The budget is now recorded in every report.

Infra bugs worth remembering: **nginx ships on :8001 in the RunPod pytorch image**, so vLLM
died with `Address already in use` while `curl -sf /v1/models` reported the port HEALTHY (nginx
answers 200 with HTML on any path) — a status-only health check would have run 500 base
episodes against a web server. Health checks now assert `"object": "list"`. Also: HF's Xet CDN
500'd mid-download (`us.aws.cdn.hf.co/xorbs/…`), fixed with `HF_TOKEN` + `HF_HUB_DISABLE_XET=1`
and a retry loop; and `RUN_SET` derived from `date` rolled over UTC midnight mid-run, which
would have split the reference and the candidates into two directories that never join.

**The generalizable rule: an empty final answer is a PARSING hypothesis before it is a model
hypothesis.** Three of the four bugs above were in code that discards the model's output.
`run_episode` now falls back to the raw turn whenever a tool-call-free turn yields no content,
so a parser quirk can never again masquerade as non-termination.

**Artifacts:** `results/eval500-20260808/{teacher-qwen3-235b,gemma-sft,gemma-base}.json` +
`README.md` (committed); 1,500 candidate traces + the reports at
`s3://restaurant-menu-corpus/v2/eval/eval500-20260808/`; the 500 eval reference traces are in
`v2/corpus.sqlite`. W&B project `menu-eval` (runs `teacher-qwen3-235b`, `gemma-sft`, `gemma-base`).

---

## 2026-08-06 — GRPO cache warm COMPLETE (902/902), after a wedged renderer + two self-inflicted hangs

The big `--modes both` GRPO warm finished: **all 902 `grpo`-split restaurants fully
covered** — 6 query templates × ≤3 URLs × {direct, browser} = **22,454 scrape rows, 0
gaps**, 41,564 total cache rows, ~199 MB, pushed to `s3://restaurant-menu-corpus/v2/cache.sqlite`.
Getting there cost two hangs, both caused by the fix for the previous one. Worth reading
before touching `src/backends.py`.

**Warm config:** `--splits grpo --queries "{name} {city} menu" "{name} menu" "{name} {city}"
"{name} {city} menu prices" "{name} {city} order online" "{name} restaurant {city} menu"
--urls-per-query 3 --modes both --workers 3 --sleep 2.0`

**First: the protections moved into the backend.** `warm_cache.py` had accumulated real
defenses (dead-end domain skips, infra/site error split, preflight) that the LIVE tool path
did not have — and worse, warm wrapped the RAW `build_scrape()` closure while `setup_tools`
slimmed before caching, so **warmed rows were stored unslimmed** and served unslimmed on
every later hit (off the SFT distribution, and junk pages froze as permanent `'ok'` because
`slim_rows` rewrites text but not `status`). All of it now lives in `build_scrape` itself:
slim-at-source, `SKIP_DOMAINS`/`SKIP_EXTENSIONS`/Content-Type sentinels, per-host throttle,
infra circuit breaker. Plus `Cache.reclassify` to repair the stale statuses.

**Hang #1 (2026-07-25) — a wedged renderer parks a worker forever.** All 3 workers idle at
0% CPU, no sockets, cache untouched for 12+ min. `py-spy dump` showed every worker inside a
Playwright protocol call. `goto`/`networkidle` have timeouts, but **`page.evaluate` (the
auto-scroll), `page.content()`, and context open/close do not** — a page whose renderer
wedges never answers. Killing the `chrome-headless-shell` processes by hand unblocked the
run in place (sentinels + pooled relaunch recovered it). Fix: `RENDER_WATCHDOG_S` (180s;
worst legit render ≈100s) kills that thread's node **driver**.

**Hang #2 (2026-08-05) — the watchdog's own zombie, and an UNKILLABLE process.** The kill
worked and the first call raised, but the dead browser stayed pooled: **`is_connected()` is
a cached flag** that only a close *event* clears, and a force-kill sends none. The next call
issued `new_context()` into a dead pipe, where Playwright's sync wrapper spins
`while not task.done(): fiber.switch()` — pinning a core **and holding the GIL**, so Ctrl+C
was never delivered (72 min of CPU burned; needed `Stop-Process -Force`). Fix:
`_forget_thread_browser()` + skip `context.close()` once fired.

**Hang #3 (2026-08-06) — poisoned threads, 585 infra failures.** `_forget_thread_browser`
skipped `pw.stop()` to dodge hang #2's hazard. But **`stop()` is the one call you must
make**: it is TRANSPORT teardown (closes the pipe; awaits a reader task already at EOF), not
a request awaiting a reply. Skipping it leaves the thread's asyncio loop running
(`run_forever` only clears the running-loop marker in its `finally`), so Playwright refuses a
second loop on that thread — **poisoning the worker for the whole run**. One wedged page per
thread poisoned all three: 585 infra failures, 5 `BrowserDeadError` aborts, warm ~89%
complete. The re-run after the fix filled all 1,498 remaining gaps.

**The generalizable rule:** against a dead Playwright driver, a **protocol** call
(`browser.close()`, `new_context()`) HANGS; a **transport** call (`pw.stop()`) is safe and
required. Conflating them cost two outages in opposite directions.

**No data was ever corrupted.** `store_if=is_cacheable` kept all 585 infra failures out of
the DB — verified 0 `asyncio loop` rows, 0 launch-failure rows. Infra failures are facts
about the machine, not the web; the URLs were simply left absent and re-fetched later.

## 2026-07-19 — first vLLM-teacher corpus build (50 sft) + context-overflow fix

Ran the 235B teacher over 50 sft restaurants (4×H100, free episodes) to calibrate cache
warming. **32 completed** (100% schema-valid, 93.8% found, mean 3.9 tool calls, 40.5 items)
+ 11 pre-existing; **7 failed on context-length 400s**.

**Tool-call calibration** (44 sft traces via `analyze_queries.py`): the Qwen teacher issues
**1.09 queries/episode** (vs the v1 Claude teacher's 2.94) — one `{name} {city} menu` (81% of
queries) and it commits. 2.7 scrapes/ep, 78% direct / 22% browser, 18.5% delivery-aggregator
(already covered by `warm_cache` `SKIP_DOMAINS`). Implication: the warm needs **fewer query
templates** than the v1 distribution implied; `{name} {city} menu` + ~3 tail templates ≈ 95%.

**Bug #10 — context overflow, not a window-size problem.** Each of the 7 failures accumulated
**81,921+ input tokens**: `openai_agent.run_episode` always requested 16384 output tokens with
no clamp, and `MAX_TOOL_CHARS=75000` ≈ 19K tokens per scrape, so a few big delivery/PDF scrapes
blew the window (even at 98304 — clamping output alone would have saved all 7). Three-layer fix:
(1) `MAX_TOOL_CHARS` 75000 → **24000** (bounds per-result growth; typical menus still fit whole);
(2) `serve_teacher.sh --max-model-len` 40960 → **131072** (headroom); (3) `run_episode` now
**clamps `max_tokens` to the remaining window and finalizes gracefully on an overflow 400**
(partial menu, not a lost trace) — the chat-path analogue of the student's
`build_gemma_completions` clamp. Unit-tested (`tests/test_openai_agent.py`, 4 new cases); to be
validated on the next build. The 32 traces are kept (idempotent build resumes past them).

## 2026-07-19 — self-hosted vLLM teacher works end-to-end: Qwen3-235B-FP8 on 4×H100

Stood up the v2 **self-hosted teacher** (replacing the Sonnet API) on vLLM/RunPod. Staged: validated the
whole tool-call path on **Qwen3-30B-A3B-Instruct-2507** (1×A100-80GB, ~$1.39/hr) first, then the real
teacher **`Qwen/Qwen3-235B-A22B-Instruct-2507-FP8`** on **4×H100-80GB** (TP=4, ~$11.96/hr). vLLM 0.25.1,
driver 580/CUDA 13 via `runpod_create.py`'s `allowedCudaVersions`.

**Repo side needed zero changes** — `build_corpus.py --teacher vllm --teacher-base-url <url> --teacher-model
teacher` already drives `openai_agent.run_episode`. This was purely a serving job. New artifacts:
[scripts/infra/serve_teacher.sh](../scripts/infra/serve_teacher.sh) (the recipe) +
[scripts/infra/smoke_teacher.py](../scripts/infra/smoke_teacher.py) (one-episode PASS/FAIL gate, exit 0/1).

**Serve flags that matter:** `--enable-auto-tool-choice --tool-call-parser hermes` (Qwen3 = Hermes tool
format), **no** `--reasoning-parser` (Instruct-2507 has no `<think>`), `--enforce-eager` (scrape-bound →
skip compile/graphs, less VRAM), `--max-model-len 40960`, `--gpu-memory-utilization 0.90`. TP=4 divides
the model's 4 KV heads cleanly.

**Bug #9 — block-scale FP8 backend, silent until first request.** vLLM 0.25.1 defaults Qwen3-FP8's dense
block-scale GEMM to FlashInfer, but the `runpod/pytorch` image has no FlashInfer cubin
(`VLLM_HAS_FLASHINFER_CUBIN=False`), no `nvcc`, no `deep_gemm`. Result: `/v1/models` returns HEALTHY, then
the **first real inference crashes the engine** — `RuntimeError: Assertion failed: !cubin.empty() ||
isPathValid(path_)` (`fp8_blockscale_gemm_sm90`), GPUs free to 0, port dies → client sees `Connection
refused`. **Fix:** `export VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0 VLLM_USE_DEEP_GEMM=0` → dense FP8 → compiled-in
**CUTLASS**, MoE → **Triton** (`ptxas`+`ninja` present). Lesson: a green `/v1/models` is not "it serves"; the
smoke (a real inference) is the gate.

**Results (self-report, `Pagliacci, Seattle`, schema-valid both):** 30B → 3 tool turns, 7 sections / 48
items; 235B → 4 tool turns (search → direct → direct → browser-escalate on its own), 8 sections / 51 items.
235B download ~150 s (~1.5 GB/s, unauthenticated HF fine); cached relaunch ready in ~80 s. Pods torn down
after validation (ephemeral; recipe is the artifact). Relaunch for the corpus build is one `runpod_create.py`
+ `serve_teacher.sh`.

## 2026-07-16 — GRPO round 1 aborted: sync tools serialize every scrape (bug #8)

First real GRPO run (H200 141 GB, `--cache-policy live`, G=8, `max_completion_length` 16384,
bs=2×accum=16). Killed after ~40 min: **step 1 never completed, GPU pinned at 0%**.

**Not stuck — serialized.** Diagnosis from the box: 6 Chromium processes, 13–17 established TCP
connections, `cache.sqlite` growing ~2.6 MB — i.e. genuinely scraping, just one call at a time.

**Root cause — TRL only parallelizes COROUTINE tools** (`grpo_trainer.py` ~1819):

    if name in sync_tool_dict:    tool_call_results.append(sync_tool_dict[name](**args))  # inline, SERIAL
    elif name in async_tool_dict: async_coros.append(...)   # -> asyncio.gather(*coros) -> PARALLEL

`build_model_tools` returns plain `def` functions, so **every live scrape blocked the whole loop**:
~**256 sequential network round-trips per step** (32 completions × up to 8 tool calls). Projected
**>40 min/step ⇒ ~100 h ⇒ ~$440** for 150 steps, nearly all of it with the GPU idle.

This is the **scrape-bound, not compute-bound** property the v2 notes predicted, in its worst form:
we bought an H200 for *memory* (real — it fixed clipping and allowed G=8), but throughput was set by
serialized network I/O, so the card idled.

**Fix (`ceb6c89`): `tools._to_async`** wraps the blocking backends in `asyncio.to_thread` so TRL sees
coroutines and gathers them. `to_thread` is the right primitive — the backends are genuinely blocking
(`requests` + **sync** Playwright) and can't be made natively async — and its threadpool reuses
threads, which fits `backends.py`'s **thread-local Chromium pool** exactly (one browser per worker
thread, reused; the same property that makes the viz server's threadpool safe).

- **Opt-in** (`setup_tools(async_tools=True)`), used only by `train_grpo`: `eval_split`,
  `gemma/agent.py` and `claude_agent` call the registry synchronously and would get coroutines back.
- **The model-facing schema is unchanged** — `functools.wraps` keeps `__name__`/`__doc__`/
  `__annotations__`, so the docstring stays defined once on the sync function. Verified
  `get_json_schema(sync) == get_json_schema(async)` for both tools: the student sees exactly the
  declarations it was SFT'd on. (`iscoroutinefunction` reads code flags, not `__wrapped__`, so it
  still reports True.)
- Measured: 4 × 0.3 s blocking calls gathered finish in ~0.3 s, not ~1.2 s.

**Lesson:** the smoke passed with `--cache-policy canned`, where tool calls are instant disk reads —
so it could not have surfaced this. **A canned smoke validates the code path but hides every I/O
property of the real run.** Cost so far: ~$4.39/hr × ~1 h of idle H200.

**Still unmeasured:** the real step time with parallel tools. Re-run before trusting any projection.

---

## 2026-07-16 — GRPO `--use-vllm` smoke PASSES (A100): 6 bugs, and the config the real run needs

First run of the **vLLM rollout path** for GRPO (the prior smoke used `--use-vllm` OFF, i.e.
transformers generation, so none of this surface was exercised). Tiny: `--max-steps 2
--num-generations 4 --limit 8`, CUDA-13 A100 80GB, colocate. **Result: `SMOKE_RC=0`, 2/2 steps,
adapter + checkpoint-2 saved.** It took **7 attempts**; each fix exposed the next layer.

**THE HEADLINE ANSWER: `skip_special_tokens` is NOT a problem on TRL's rollout path.**
`tools/call_frequency: 151.2`, `tools/failure_frequency: 0` — tool calls fire and parse. This was
the top risk (it would have silently zeroed grounding and looked like a bad model). TRL's tool loop
uses the Gemma tokenizer's built-in `response_schema`, so it sees the markers. Cleared.

**Bugs found (all invisible until it ran; all fixed + pushed):**

| # | failure | cause |
|---|---|---|
| 4 | `OSError: Can't load feature extractor for '/workspace/merged'` | **TRL colocate loads vLLM from the model PATH on disk** (not by syncing the in-memory HF policy — I had assumed the latter). So the `preprocessor_config.json` gap applies to GRPO exactly as to `vllm serve`. Fix: point `--model-path` at `merged-text`. |
| 5 | `ValueError: Target modules .*language_model\..*$ not found` | Fixing #4 exposed it: the LoRA regex assumes the multimodal tree (it exists to exclude the towers PEFT can't adapt), but `to_text_only` DROPS the towers, so text-only modules are plain `model.layers.N.`. Fix: derive targets from the model's real module names. |
| 6 | `TypeError: <lambda>() takes 2 positional arguments but 3 were given` | `model.py`'s SDPA GQA patch hardcoded transformers 5.10.x's `use_gqa_in_sdpa(attention_mask, key)`. transformers **5.14.1** (pulled by vLLM into the GRPO env) calls it with 3. Fires mid-training at the first forward. Fix: `*args, **kwargs` — it answers False unconditionally anyway. |
| 7 | `OutOfMemoryError: Tried to allocate 32.00 GiB` | Gemma 4's **vocab_size=262144** ⇒ fp32 logits = `bs × max_completion_length × vocab × 4B` = `4 × 8192 × 262144 × 4B` = **34.4 GB**, matching the alloc exactly. Our guard forced `per_device_bs` to be a *multiple* of G; TRL's real rule is that the **effective** batch (`bs × accum × num_processes`) be divisible by G — TRL computes group advantages over the whole generation batch and only chunks fwd/bwd, so **`per_device_bs` is a pure memory knob**. Fix: match TRL's rule ⇒ bs=1 legal ⇒ logits 8.6 GB. |

Also learned: **`vllm_gpu_memory_utilization` is NOT a memory lever** — it's a fraction of TOTAL VRAM
covering vLLM's *weights + KV*, so below ~0.19 (15 GB weights / 79 GB) vLLM has no room for KV blocks
and dies with `No available memory for the cache blocks`. Floor is ~0.28 in practice.

**Passing config (A100 80GB):** `--per-device-train-batch-size 1 --gradient-accumulation-steps 16
--vllm-gpu-memory-utilization 0.30`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (the last OOM
showed 8.04 GiB reserved-but-unallocated fragmentation).

**Smoke metrics — plumbing only, DO NOT read as quality:**

    tools/call_frequency 151.2   tools/failure_frequency 0
    rewards/structure/mean 0.75  rewards/found/mean 0.25  rewards/grounding/mean -0.2125
    completions/mean_length 5161  max 7367  clipped_ratio 0.375
    step_time 343.6s

- **grounding is NEGATIVE (-0.21) because the smoke ran `--cache-policy canned`** — which starves
  grounding *by design* (reward.py and train_grpo's default both say `live`): a canned MISS returns a
  constant, so anything the student explored that the teacher didn't yields junk evidence, and its
  items can't ground. My smoke chose canned for speed/offline determinism. **The real run MUST use
  `live`** or the reward measures nothing.
- **`clipped_ratio 0.375` is a REAL problem for the real run:** 37.5% of rollouts hit
  `max_completion_length=8192` and were truncated. TRL appends tool responses *into* the completion,
  and scraped pages are large, so 8192 is too small for an 8-tool-call episode. Truncated ⇒ no final
  JSON ⇒ that likely explains most of `found=0.25`.

**Why the real run should move to an H200 (141 GB), not the A100:**
1. `clipped_ratio 0.375` ⇒ need `max_completion_length` ≫ 8192 ⇒ more logits memory (16384 @ bs=1 =
   17.2 GB).
2. `step_time 343.6s` ⇒ **23.7 h/epoch** (249 steps @ 4 prompts/step over 995 prompts) ≈ $33/epoch on
   A100. A bigger KV cache (util 0.3 of 141 GB = 42 GB vs 24 GB) buys rollout concurrency.
3. **G=4 is minimal** for group-relative advantage (`frac_reward_zero_std: 0.25` — a quarter of groups
   had zero reward variance, i.e. no learning signal from them). Headroom would allow G=8.
On the A100 every knob is spent: vLLM ~24 GB + policy 15 GB + logits 8.6 GB, bs already at 1 and
`vllm_gpu_memory_utilization` already at its floor.

**Artifacts:** pod-local (smoke adapter discarded — it's 2 steps of a starved reward).

---

## 2026-07-16 — v1 SFT student, FULL 500-episode eval, bf16 via vLLM (the real headline)

**First clean full-500 eval of the v1 student served correctly** (merged bf16, text-only, on vLLM
— not 4-bit, not a 50-ep subset). Same seed-42 plan as every prior run (300 free + 200
conditioned, `--conditioned-frac 0.4`, student prompt, `--cache-policy live`). Served the
backfilled text-only checkpoint on a CUDA-13 A100 at `--max-model-len 98304`, `--workers 16`.
**500/500 completed, 0 failed, 0 context-400s.**

Self-report (no reference):

| slice | schema-valid | found=true | mean items | mean sections | price cov |
|---|---|---|---|---|---|
| **all (n=500)** | **83.4%** | **78.8%** | 24.0 | 4.11 | 0.728 |
| free (n=300) | 83.3% | 78.3% | 29.3 | 4.66 | 0.730 |
| conditioned (n=200) | 83.5% | 79.5% | 16.1 | 3.29 | 0.726 |

**The headline: 78.8% found on the full 500, up from 24% on the 2026-07-14 4-bit full-500.** Almost
all of that gain is the serving fix (bf16 not 4-bit) + the `_ALWAYS_ANSWER_RULE` termination prompt,
both already established on subsets; this is the first time they're confirmed at full scale on both
splits. The 24%→79% arc is the whole "the v1 student was never as bad as it looked" story landing.

**The surprise — conditioned no longer collapses.** Every prior read (bf16 subset: 70% free vs 35%
conditioned) said dietary filtering was the big v2 quality lever. At full scale that gap **is gone**:
conditioned found=true 79.5% ≈ free 78.3%, and schema-validity is identical across slices (83.5 vs
83.3). Re-reading [[v1-sft-failure-mode]]: the "conditioned is the hard part" finding was itself
mostly a *termination* artifact on a 20-episode slice (7/20→17/20 under the new prompt). With
termination fixed and n=200, conditioned success matches free. **What conditioning still costs is
menu SIZE, not success**: conditioned mean items 16.1 vs free 29.3 — expected and correct, since a
dietary filter legitimately removes items. So the model finds and filters fine; it just returns a
(correctly) smaller menu. This meaningfully weakens the case for a separate dietary-judge model as a
v2 priority.

**Caveats:** self-report only (schema-valid + non-empty + item/price counts), NOT scored against a
teacher reference, so "found=true" ≠ "menu is correct" — it means well-formed and non-empty.
`price_coverage 0.73` is a soft quality signal (share of items carrying a price). A reference-scored
pass (`--reference`) is the next fidelity step if we want a precision/recall number.

**Throughput:** 500 episodes in **~34 min** (~15 eps/min, 16 workers) vs the HF `--workers 1` path's
**5.4 h** for the same 500 — **~9–10× faster**. This is the entire reason the vLLM serving work
mattered: full-500 evals are now a coffee break, not an afternoon.

**Three silent bugs surfaced getting here**, all invisible under HF, all only findable by actually
serving (details in the 2026-07-16 vLLM entry below): (1) 54 missing KV-shared tensors; (2)
`skip_special_tokens=True` eating the tool protocol; (3) unclamped `max_tokens` 400ing long episodes.
Bug 3 was caught *during this eval's first 3 episodes* — before the fix it would have scored the
longest, most-gathered episodes as FAILED, biasing the headline DOWN.

**Artifacts:** `v1/eval/20260716/gemma_bf16_vllm_500/{report.json, candidates.tgz, eval.log}`
(all with `x-amz-meta-md5`). Merged checkpoint md5 now also stamped:
`66145fec9549e32682c2a426dbf4739f`.

---

## 2026-07-16 — vLLM serving: the three real blockers, all found and fixed

Ran the whole path on a **CUDA-13 A100 80GB PCIe** ($1.39/hr): provision → pull merged
checkpoint → text-only convert → `vllm serve`. Three distinct blockers, each of which had been
misattributed before. **None of them was `head_dim=512`.**

**1. The driver wall was a PROVISIONING bug, not a hardware fact.** We knew we needed driver
≥580 (vLLM ≥0.20 = Gemma-4 support = torch cu130) but kept landing on CUDA-12.8 hosts. Cause:
**`runpodctl create pod` has no CUDA/driver flag** ([runpodctl#253](https://github.com/runpod/runpodctl/issues/253)),
so the CLI cannot express the constraint at all. The **REST API** (`POST rest.runpod.io/v1/pods`)
takes **`allowedCudaVersions: ["13.0"]`**. With it, the *same* `cu1281` image booted on
**driver 580.126.20 / CUDA 13.0** — proving the image never mattered; only the host filter does.
Now wrapped in [scripts/infra/runpod_create.py](../scripts/infra/runpod_create.py).

**2. FA4 is Hopper-only — but that is NOT fatal.** The H200 logged `Using FA4 for all layers`;
the A100 logs:

    Gemma4 model has heterogeneous head dimensions (head_dim=256, global_head_dim=512).
    FA4 not available, forcing TRITON_ATTN backend.

vLLM **falls back cleanly to TRITON_ATTN** and still handles the mixed 256/512 heads. So Ampere
is viable for Gemma-4 on vLLM — relevant because A100 ($1.39) vs H200 ($3.59) is the GRPO-rollout
cost question. (Throughput on TRITON_ATTN vs FA4 is unmeasured; that's the open question, not
whether it runs.)

**3. THE REAL BUG — our merged checkpoint is missing 54 tensors, and it's our pipeline.**
`vllm serve` died with:

    ValueError: Following weights were not initialized from checkpoint:
    {'model.layers.24..41.self_attn.k_norm.weight', ...}   # 18 layers

Diffing the base checkpoint's safetensors header (via an S3 **range request** — no 16 GB download)
against our merged one:

| checkpoint | tensors | `k_norm` layers | `q_norm` layers |
|---|---|---|---|
| base `gemma-4-E4B-it` | **2130** | **0–41** | 0–41 |
| our `merged` (SFT) | **2076** | **0–23** | 0–41 |

Exactly **54 missing = 18 × {k_norm, k_proj, v_proj}** for layers 24–41, zero extras. Root cause:
Gemma-4 E4B sets **`num_kv_shared_layers=18`**, so its last 18 of 42 layers reuse K/V from an
earlier layer and never compute their own. **transformers honors this and never instantiates
those params** → they are dropped as unexpected when loading the base → absent from every
checkpoint we save since. This is **invisible under transformers** (`missing=0 unexpected=0`, and
our 88% HF eval is unaffected — they really are dead weights). **vLLM's Gemma4 builds a fused
`qkv_proj` + `k_norm` for every layer** and hard-fails if they're absent, even though its shared
layers discard the K/V they compute.

**Fix:** `to_text_only.py --base <base_dir>` copies those 54 tensors raw out of the base
safetensors and injects them after `save_pretrained` (they can't come from `state_dict()` — the
params don't exist on the model). Values are irrelevant (discarded downstream); existence is what
vLLM demands. **Do NOT** instead disable vLLM's `enable_weights_track` check: that swaps a clean
error for uninitialized memory.

**`--base` does not need the full 16 GB.** safetensors keeps a JSON header at byte 0 with every
tensor's `data_offsets`, so you can read the key list and then **range-GET only the tensors you
want**. The 54 KV-shared tensors are **110 MB** (`k_norm` is 512 B; `k_proj`/`v_proj` are
[512, 2560] bf16 ≈ 2.6 MB each) — a **145× smaller** pull than the whole checkpoint, and the
mini-shard drops straight into `--base` since the backfill just globs `*.safetensors`. Worth
remembering generally: pulling a whole checkpoint to inspect or borrow a few tensors is
almost never necessary.

**4. `ninja` must be on `PATH` (Ampere only).** TRITON_ATTN JIT-compiles kernels and shells out to
**`ninja` by name**. Launching `/opt/vllm-env/bin/vllm serve` by absolute path does NOT activate the
venv, so `PATH` lacks `/opt/vllm-env/bin` and the engine dies **late** — after weights load and CUDA
graphs capture — with `FileNotFoundError: [Errno 2] No such file or directory: 'ninja'`, even though
`ninja` is installed right beside the `vllm` binary. Launch via `env PATH=/opt/vllm-env/bin:$PATH`.
Hopper/FA4 never compiles Triton kernels and never hits this.

**5. THE SECOND REAL BUG — `skip_special_tokens=False` is load-bearing on the vLLM student path.**
Gemma's tool protocol *is* special tokens, and vLLM's detokenizer defaults to stripping them.
`build_gemma_completions` didn't request otherwise, so every vLLM rollout would have **silently**
looked like a broken model:

| `skip_special_tokens` | text | `stop_reason` | `finish` |
|---|---|---|---|
| `True` (vLLM default) | `call:web_search{query:...}` ×N, no markers | `None` | `length` |
| `False` | `<\|tool_call>call:web_search{query:<\|"\|>...<\|"\|>}` | `<tool_call\|>` | `stop` |

With markers stripped, `parse_response` sees plain `content` and **zero tool calls** (the agent loop
never fires), and the `<tool_call\|>` stop string can never match — so generation rambles to
`max_tokens` *and* the stop-marker re-append is dead code. CLAUDE.md already carried this rule
("decode with `skip_special_tokens=False`") but it had only ever been applied to the **HF** path;
it has to be requested explicitly over HTTP. Fixed in [src/serving/openai_agent.py](../src/serving/openai_agent.py).

**VERIFIED END-TO-END** through the real `build_gemma_completions` under the real student prompt:

    RAW:    <|tool_call>call:web_search{query:<|"|>Kashish Indian Curry Kirkland menu<|"|>}<tool_call|>
    PARSED: {'role','tool_calls'} -> web_search(query="Kashish Indian Curry Kirkland menu")

Serving stats (A100 80GB, `--max-model-len 40960 --gpu-memory-utilization 0.85`): model load
**14.23 GiB / 3.8 s**, torch.compile 47 s (cached after), **KV cache 2,695,181 tokens →
65.8× concurrency** at 40k ctx. That concurrency is the whole point vs the HF path's `--workers 1`.

**Takeaways:** (a) `to_text_only.py`'s old `missing=0 unexpected=0` was a **false all-clear** — it
only proves transformers' *own* expectations were met, not that the checkpoint is complete;
(b) any future consumer that expects the full base key set will hit this same gap, so the merge
in `train_sft.py` is lossy by construction — the backfill is the seam that repairs it;
(c) **both real bugs were silent** — one a hard crash only vLLM could surface, one a
plausible-looking degenerate output. Neither was reachable without actually serving the thing.

**Artifacts:** pod-local only (nothing pushed to S3 — the text-only checkpoint is cheap to
rebuild from `merged` + the 110 MB mini-shard, ~6 min).

---

## 2026-07-14 — Gemma v1 SFT student, full eval (n=500, 4-bit serving)

**Model:** `v1/models/gemma-menu-sft-20260714/merged` (the v1 SFT LoRA, merged to bf16), loaded
for inference **in 4-bit nf4** (`eval_split.py` → `load_model(quantize=True)`).
**Run:** `scripts/eval_split.py --model gemma --model-path models/merged --limit 500
--conditioned-frac 0.4 --seed 42 --cache-policy live` on 1× A100 80GB (RunPod). 500/500, 0 failures,
~5.4 h.

Self-report:

| split | schema-valid | found=true | mean items |
|---|---|---|---|
| all (500) | 25.0% | 24.2% | 8.8 |
| free (300) | 24.3% | 23.7% | 11.3 |
| conditioned (200) | 26.0% | 25.0% | 5.2 |

**Failure-mode breakdown** (the important part — categorizing all 500 candidates):

| outcome | count | % |
|---|---|---|
| empty output (returned nothing; `extract_json` sees `char 0`) | 362 | 72.4% |
| malformed / truncated JSON | ~13 | 2.6% |
| valid JSON but `found=false` | 4 | 0.8% |
| **success** (schema-valid + found) | 121 | 24.2% |

**When it succeeds, menus are teacher-quality:** mean **36.5 items** (median 27) vs Claude's 39.0.

**Takeaway:** this is **not** an extraction-quality ceiling. The 4B, given the same tools/prompt/schema,
produces roughly Sonnet-quality menus *when it finishes*. The dominant failure (72%) is an **empty final
answer** — the agent burns its 8-tool-call budget without committing to emit the JSON (`agent.py` returns
`""`). So the gap is **agentic-loop reliability / termination**, the expected weak axis for a small model —
and precisely what GRPO's completeness reward and better/more SFT target. Open follow-ups: (1) does bf16
serving cut the empty-output rate? (see pending entry); (2) does the student under the *teacher* prompt
complete more often (distillation gap vs. capability)?

**Artifacts:** `v1/eval/20260714/gemma/{report.json, candidates.tgz, eval_full.log}`

---

## 2026-07-14 — Claude Sonnet baseline eval (n=500)

**Model:** `claude-sonnet-5` (teacher/frontier reference), same eval plan + student prompt, run from the
devbox (thread-parallel).

Self-report:

| split | schema-valid | found=true | mean items |
|---|---|---|---|
| all (500) | 99.4% | 95.0% | 39.0 |
| free (300) | — | 96.7% | 49.3 |
| conditioned (200) | — | 92.5% | 23.5 |

**Takeaway:** the task is clearly doable with these exact tools/prompt/schema — Sonnet nearly always
finds and emits a rich valid menu. This is the ceiling the student is measured against; the conditioned
slice is lower-item by design (dietary filtering removes items).

**Artifacts:** `v1/eval/20260714/claude/{report.json, candidates.tgz}`

---

## 2026-07-14 — Gemma v1 SFT training run (H200)

**Base:** `google/gemma-4-E4B-it` (multimodal `Gemma4ForConditionalGeneration`; ~8B total, ~4B-effective
text path). **LoRA** r16 / α32, scoped by regex to the **`language_model`** submodules only (PEFT can't
adapt the towers' `Gemma4ClippableLinear`) → 258 modules, ~34.9M trainable params.
**Data:** `v1/sft/train.jsonl` (948 examples; **786 kept at `--max-length 32768`**, ~83%), eval-frac 0.05.
**Trainer:** `scripts/train/train_sft.py`, 3 epochs, per-device batch 1 × grad-accum 8, `attn=sdpa`, bf16.
**Hardware:** 1× H200 (RunPod). Single-run, no length ramp (per decision to start at 32k).

**Bound diagnosis:** **memory-bound, not param-bound.** Peak memory is dominated by the training loss
logits — `seq_len × vocab(262,144) × 4B fp32` — not the 34.9M LoRA params. At 32k this is tens of GB of
transient logits; an 80GB H100 OOMed, which is why training moved to the H200. (A chunked/fused
cross-entropy that recomputes logits per sequence-chunk in backward is prototyped in scratch but not yet
integrated — the real lever if we push context or batch further.)

**Artifacts:** `v1/models/gemma-menu-sft-20260714/{adapter, merged}` (merged `model.safetensors` ≈ 15.9 GB).

---

## 2026-07-15 — Prompt fix for termination: `_ALWAYS_ANSWER_RULE` (paired n=50) ✅

**Hypothesis:** the residual empty-output failure that survived the bf16 fix (~35-40% empty) is
fixable by PROMPTING — tell the student, as a standing system-prompt rule, that every episode must
end with a JSON object. Implemented as the shared `_ALWAYS_ANSWER_RULE` in `src/prompts.py`
(branch `prompt-termination`): never end with an empty reply; the moment you can't call more tools,
emit the menu from what you have; a partial menu beats no reply; else the found=false shape.

**Setup — a clean A/B.** Same 50 episodes (seed 42, 30 free + 20 conditioned), same merged v1 SFT
student, **same HF bf16 stack**, same warm cache, same code path as the 2026-07-14 bf16 subset below.
**Only the prompt differs.** (Run on an H200; the baseline ran on an A100 — the only uncontrolled
variable, immaterial at temperature 0.)

Self-report:

| split | old prompt | **new prompt** |
|---|---|---|
| all schema-valid | 62.0% | **88.0%** |
| all found | 56.0% | **88.0%** |
| free found (30) | 70.0% | **90.0%** |
| conditioned found (20) | 35.0% | **85.0%** |

**Paired on the 50 identical episodes:**

| outcome | old prompt | **new prompt** |
|---|---|---|
| success | 28 (56%) | **44 (88%)** |
| empty | 19 (38%) | **6 (12%)** |
| free | 21/30 | **27/30** |
| conditioned | 7/20 | **17/20** |

**16 episodes fixed (empty→success), 0 regressed.** Monotone improvement.

**The one number that needed scrutiny — and its resolution.** Mean items *when successful* fell
31.9 → 22.7, which would be alarming if the rule were causing PREMATURE termination (stop searching
too early → thinner menus). It isn't — it's pure composition:
- Episodes successful under BOTH prompts (n=28): items **31.9 → 30.9**, with 20/28 within ±2
  (3 bigger, 5 smaller). The already-working episodes are **untouched**.
- The 16 RESCUED episodes (previously returned *nothing*) average **8.2 items** — thin but real,
  exactly what the rule asks for. They drag the mean down while strictly adding value.

**Takeaways:**
1. **The v1 student's dominant failure was largely a missing instruction, not a missing capability.**
   It obeys a termination rule it was never SFT'd on — so "the behaviour didn't distill" was the wrong
   diagnosis; nothing in the shipped student prompt ever told it to commit. NOTE this contradicts
   CLAUDE.md's guidance ("add contrastive data rather than leaking guidance into the student prompt")
   for THIS failure mode — because termination is a task-completion requirement, not distillable strategy.
2. **Conditioned episodes were mostly a termination failure, not a dietary-filtering failure**
   (7/20 → 17/20). The "dietary filtering is the hard part" read from 2026-07-14 was largely an
   artifact of empty outputs; the real filtering gap is much smaller than it looked.
3. **This de-risks GRPO**: a policy that reliably emits an answer gives GRPO real reward signal to
   optimize, instead of a wall of zero-reward empty rollouts.

**Caveat:** measured on the v1 checkpoint under a prompt it wasn't trained with (train/inference
mismatch). A future SFT run should render its data under this same student prompt so train == inference.

**Artifacts:** local scratch (S3 push pending — the dev-box AWS session expired). Intended:
`v1/eval/20260715/gemma_bf16_newprompt/{report.json, candidates.tgz, eval.log}`

---

## 2026-07-15 — vLLM serving: what actually blocks it (head_dim was a red herring)

Findings from trying to serve the merged student on vLLM (H200), which resolve a risk the vLLM design
doc has flagged since day one:

1. **`head_dim=512` is a NON-ISSUE.** vLLM logs:
   `Gemma4 model has heterogeneous head dimensions (head_dim=256, global_head_dim=512). Using FA4 for
   all layers to avoid mixed FA3/FA4 penalty.` It recognises Gemma-4's mixed heads and handles them
   natively. `notes/vllm_inference.html`'s central caveat ("validate on Hopper; may fail on Ampere")
   is **obsolete**.
2. **The real requirement: vLLM ≥0.20 AND driver ≥580 (CUDA 13).** Gemma-4 support and cu130 torch
   arrived together, so no version pin escapes the driver requirement:

   | vLLM | torch | gemma4 |
   |---|---|---|
   | 0.12.0 | 2.9.0+cu128 | ✗ |
   | 0.18.1 | 2.10.0+cu128 | ✗ |
   | 0.20.2 / 0.21.0 / 0.22.1 / 0.25.1 | 2.11.0+**cu130** | ✓ |

   The driver is a **host** property (RunPod images inherit it) — a CUDA-12.8 host cannot serve
   Gemma-4 on vLLM at any vLLM version. Pick a host with driver ≥580.
3. **Our merged checkpoints can't be served on vLLM's multimodal path at all**:
   `train_sft._save_outputs` saves model+tokenizer only, so there is no `preprocessor_config.json`
   and vLLM's `AutoProcessor` dies. Neither the S3 base copy nor the HF cache has that file.
   **Fix: `scripts/train/to_text_only.py`** rebuilds a text-only `Gemma4ForCausalLM` from the multimodal
   checkpoint (verified `missing=0 unexpected=0`, 7.46B params, towers dropped) — which is what the
   design doc wanted anyway ("we never use the vision/audio towers") and needs no processor.

---

## 2026-07-14 — Gemma v1 SFT student, bf16 subset (n=50, paired vs 4-bit)

**Why:** the full eval above served the merged bf16 checkpoint **re-quantized to 4-bit nf4**. This tests
whether **bf16 serving** raises the completion rate (fewer empty outputs) — a serving artifact vs. a
training/termination problem. Same seed-42 plan, `--limit 50` (30 free + 20 conditioned), warm cache,
`EVAL_BF16=1` env switch added to `eval_split.py` (`load_model(quantize=…)`). bf16 decode is ~2–3× slower
than 4-bit (memory-bandwidth-bound: 16 GB weights/token vs 4 GB), ~110 s+/episode.

Self-report (n=50):

| split | schema-valid | found | mean items | (4-bit full-500 for ref) |
|---|---|---|---|---|
| all (50) | **62.0%** | **56.0%** | 17.9 | 25.0 / 24.2 / 8.8 |
| free (30) | **76.7%** | **70.0%** | 24.1 | 24.3 / 23.7 / 11.3 |
| conditioned (20) | 40.0% | 35.0% | — | 26.0 / 25.0 / 5.2 |

**Paired on all 50 identical episodes (same restaurants, both serving modes):**

| outcome | 4-bit | **bf16** |
|---|---|---|
| success | 11 (22%) | **28 (56%)** |
| empty | 37 | **19** |
| valid-not-found | 2 | 3 |

**21 episodes fixed by bf16** (empty→success), 4 regressed (greedy-decode noise), **net +17**. The lift is
concentrated in the **free** split (6→21 success, 20%→70%); **conditioned barely moved** (5→7, 25%→35%).
Item counts when both succeed are comparable — bf16 changes *completion rate*, not menu quality.

**Takeaway:** the 4-bit **serving** path badly understated the student. Much of the "empty output" failure
was nf4 quantization derailing the agentic loop, not the model's ceiling — the true free-split success is
~70%, not ~24%. Two distinct residual gaps remain: (1) ~19/50 still empty under bf16 = the real
SFT/termination gap (GRPO target); (2) **conditioned/dietary filtering is the genuinely hard part** (35% vs
70% free) — the biggest single quality lever for v2. **Action items:** always serve/eval the student in
bf16 (or vLLM on the merged bf16 model — `notes/vllm_inference.html`); a full-500 bf16 eval is still worth
doing for a clean headline number (slow single-worker here → do it via the vLLM path).

**Artifacts:** `v1/eval/20260714/gemma_bf16/{report.json, candidates.tgz, eval_bf16.log}`
