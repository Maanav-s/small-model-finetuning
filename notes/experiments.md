# Experiment log

A running, append-only record of training/eval runs and their results. Newest entries
at the top. Each entry: what ran, the config that matters, the numbers, the takeaway, and
where the artifacts live (S3 is the source of truth — pods are ephemeral).

Conventions:
- **Eval plan** is the seed-reproducible mix from `scripts/build_corpus.py` (`load_seeded_rows` →
  `plan_episodes`): `--split eval --seed 42 --conditioned-frac 0.4`, rendered with the **student**
  prompt (what we ship). Same plan across models → per-episode filenames line up.
- **Self-report** metrics (no teacher reference): `schema-valid %`, `found=true %`, `mean items`.
- S3 bucket `restaurant-menu-corpus`, prefix `v1`, region `us-west-2`.

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
Now wrapped in [scripts/runpod_create.py](../scripts/runpod_create.py).

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
**Trainer:** `scripts/train_sft.py`, 3 epochs, per-device batch 1 × grad-accum 8, `attn=sdpa`, bf16.
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
   **Fix: `scripts/to_text_only.py`** rebuilds a text-only `Gemma4ForCausalLM` from the multimodal
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
