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
