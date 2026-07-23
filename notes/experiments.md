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
