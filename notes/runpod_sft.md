# RunPod SFT run — Gemma 4 E4B menu extractor (2× H200)

Setup + run guide for the LoRA SFT stage ([scripts/train/train_sft.py](../scripts/train/train_sft.py))
on a **RunPod pod with 2× NVIDIA H200 (141 GB each, Hopper / sm_90)**. This is the
distillation step: the student learns the teacher's trajectories re-rendered under
the **student** prompt (context distillation — see [CLAUDE.md](../CLAUDE.md)). Data
comes from `data/sft/<family>/train.jsonl` ([scripts/datasets/build_sft.py](../scripts/datasets/build_sft.py),
family `gemma-4-e4b-it`).

Unlike the dev box (1× 24 GB, ~15 GB host RAM, no swap), the H200 pod is not
memory-bound: we load the base in **bf16** (not 4-bit) and LoRA-train it. A 4B +
LoRA fits in one H200 many times over, so we use plain **DDP** (one process per
GPU, gradients all-reduced) — no DeepSpeed/FSDP.

> **This is the HF-training path, not the vLLM path.** SFT uses `transformers` +
> `trl` + `peft` on the torch `cu128` wheel (`uv sync`). The `cu130` / driver-580
> requirement in CLAUDE.md is a **vLLM-serving** constraint (eval / GRPO rollouts),
> not a training one. Do not conflate them.

---

## 1. Create the pod

- **Secure Cloud**, 2× H200 template.
- **Attach a network volume AT CREATION**, mounted at `/workspace`. Only the
  network volume survives pod termination — the container disk does **not**.
  **All checkpoints must land under `/workspace`.**
- Template **environment variables** (set them here AND in `~/.bashrc`, step 3 —
  full-SSH sessions sometimes don't inherit template env):
  - `HF_HOME=/workspace/hf` — cache the gated weights on the persistent volume so
    you download them once.
  - `HF_TOKEN=<token>` — a HuggingFace token whose account has **accepted the
    google/gemma-4-E4B-it license** (the model is gated; the load 403s otherwise).
  - The S3 config for pulling the dataset / pushing outputs (`S3_BUCKET`,
    `S3_PREFIX=v2`, `AWS_DEFAULT_REGION`, and a **scoped** key — see
    [S3_setup.md](S3_setup.md); never put root creds on the pod).

After boot, verify GPUs and topology:

```bash
nvidia-smi
nvidia-smi topo -m     # NVLink is nice-to-have, not required; PCIe is fine for a 4B all-reduce
```

You should see **2** GPUs. If only 1 shows, the pod isn't 2-GPU — DDP will silently
run single-process.

---

## 2. Get the code + data

```bash
cd /workspace
git clone <this-repo> small-model-finetuning
cd small-model-finetuning
```

Bring the SFT dataset onto the pod (it is git-ignored; source of truth is S3):

```bash
# option A: pull just the SFT export from S3 (needs AWS creds on the pod)
uv run python scripts/infra/corpus_sync.py pull --only sft
#   -> data/sft/gemma-4-e4b-it/train.jsonl (+ .meta.json)

# option B: rebuild it from the corpus on the pod (also needs the corpus DB)
uv run python scripts/infra/corpus_sync.py pull --only corpus.sqlite
uv run python scripts/datasets/build_sft.py            # corpus sft split -> data/sft/<family>/train.jsonl

# option C: just scp data/sft/gemma-4-e4b-it/train.jsonl up if you built it elsewhere
```

`train_sft.py` only needs `data/sft/<family>/train.jsonl`. It does **no** scraping,
so you do **not** need `playwright install` on the pod. The base weights download
from the HF hub via `HF_TOKEN` (cached under `HF_HOME`); a copy also lives at
`s3://$S3_BUCKET/v2/models/gemma-4-e4b-it/base/` if you prefer to pull it.

---

## 3. Install deps

```bash
apt-get update && apt-get install -y tmux

# env (mirror the template vars into the shell too)
echo 'export HF_HOME=/workspace/hf'   >> ~/.bashrc
echo 'export HF_TOKEN=<token>'        >> ~/.bashrc
source ~/.bashrc
hf auth login   # optional alternative to HF_TOKEN; the account must have accepted the gemma license

# python env — uv is the package manager (never bare pip)
uv sync         # reproduces pyproject.toml + uv.lock, incl. trl / peft / accelerate / torch cu128
```

Sanity-check the versions match what the trainer was written against:

```bash
uv run python -c "import trl,peft,accelerate,transformers,torch; \
print('trl',trl.__version__,'peft',peft.__version__,'accelerate',accelerate.__version__, \
'transformers',transformers.__version__,'torch',torch.__version__)"
# expected (or newer, verify): trl 1.5.1  peft 0.19.1  accelerate 1.13.0  transformers 5.10.2  torch 2.11.0+cu128
```

---

## 4. Dry-run first (cheap, catches data/config errors before burning GPU time)

Runs on CPU/tokenizer-only — builds the dataset + resolved config, prints the
token-length distribution, the drop-for-length count, and one rendered+masked
example. **Do this once** to confirm the length distribution and pick `--max-length`.

```bash
uv run python scripts/train/train_sft.py --dry-run
```

Read the printed **length p50/p90/p95/p99/max**. Examples longer than `--max-length`
are **dropped** (never right-truncated — that would cut off the final-JSON target).

> **Measured on the current dataset (2074 examples, `MAX_TOOL_CHARS=24000`):**
> **p50 ≈ 9.3k, p90 ≈ 37k, p95 ≈ 56k, p99 ≈ 97k, max ≈ 116k tokens** (episodes carry
> several capped scraped pages). At the trainer default **`--max-length 32768` →
> keep 1834 / 2074 (88%), drop 240 (12%)** — the dropped rows are the longest,
> many-scrape episodes. Raising to ~49k keeps ~93%, ~65k keeps ~96%, but the LM-head
> loss logits grow with sequence length (`seq × vocab(262k)`, upcast to fp32) and a
> ~65k sequence is ~69 GB of logits alone — fits an H200 but watch the first steps.
>
> The ~12% drop at the default is the intended trade-off; the recipe deliberately
> does **not** chase p95 into OOM territory. **Do NOT lower `MAX_TOOL_CHARS` below
> 24000 to shrink episodes** — 24000 is what the teacher saw when it produced each
> menu, so truncating below it would cut the very text the target was grounded in
> (and 24000 is already the teacher-reliability ceiling; see [CLAUDE.md](../CLAUDE.md)).

---

## 5. Launch training (under tmux)

tmux survives an SSH drop (it does **not** survive a pod restart — that's why
checkpoints go to `/workspace`).

```bash
tmux new -s sft
cd /workspace/small-model-finetuning

accelerate launch --config_file configs/accelerate_ddp.yaml \
    scripts/train/train_sft.py \
    --output-dir /workspace/gemma-menu-sft \
    --num-train-epochs 3 \
    --lora-r 16 --lora-alpha 32 \
    --learning-rate 2e-4
    # --max-length defaults to 32768; --data defaults to data/sft/<family>/train.jsonl
```

Detach with `Ctrl-b d`; re-attach with `tmux attach -t sft`.

**torchrun equivalent** (same result, if you prefer it over accelerate):

```bash
torchrun --nproc_per_node=2 scripts/train/train_sft.py --output-dir /workspace/gemma-menu-sft
```

> **You MUST use accelerate/torchrun.** Plain `python scripts/train/train_sft.py`
> launches a **single** process — it trains on GPU 0 only and leaves GPU 1 idle (no DDP).

### Effective batch size

`effective batch = per_device_train_batch_size × gradient_accumulation_steps × num_GPUs`.
With the defaults: **1 × 8 × 2 = 16**. The trainer reads `WORLD_SIZE` (set by
accelerate/torchrun) for the GPU count, so the printed math is correct under the
launcher. To change the effective batch, adjust `--gradient-accumulation-steps`
(memory-free) rather than `--per-device-train-batch-size` (H200 has headroom, but
long sequences dominate activation memory even with gradient checkpointing on).

---

## 6. Outputs

Written under `--output-dir` (on `/workspace`), by [scripts/train/train_sft.py](../scripts/train/train_sft.py):

- `…/adapter/` — the LoRA adapter (`save_pretrained`) + tokenizer.
- `…/merged/`  — the adapter merged into the bf16 base (`merge_and_unload`) +
  tokenizer, a **plain HF model** so the eval runner needs no peft:

  ```bash
  uv run python scripts/eval/eval.py --model gemma --model-path /workspace/gemma-menu-sft/merged
  ```

- `…/meta.json` — the lineage record (base + dataset by md5, LoRA/quant recipe,
  hyperparams, git sha) written by [src/run_meta.py](../src/run_meta.py).

The merge runs on the main process only (after `wait_for_everyone`), reloading the
base to keep it DDP-wrapper-free.

**Back the outputs up to S3 before terminating.** The v2 convention (see CLAUDE.md)
is to store the **base + adapter only** under
`v2/models/gemma-4-e4b-it/sft/<run-id>/` and **regenerate** `merged` / `merged-text`
on demand — the merged bf16 checkpoint is large and reproducible from base+adapter.

> **To serve the checkpoint on vLLM** (eval at scale / GRPO rollouts), you must first
> rebuild it as a text-only class with the KV-shared tensors backfilled — vLLM cannot
> load the raw `merged` dir (missing `preprocessor_config.json`; 54 uninitialized
> `k_norm`/`k_proj`/`v_proj` tensors from `num_kv_shared_layers=18`):
> ```bash
> uv run python scripts/train/to_text_only.py /workspace/gemma-menu-sft/merged \
>     /workspace/gemma-menu-sft/merged-text --base /workspace/base/gemma-4-E4B-it
> ```
> See CLAUDE.md "vLLM serving" for the full story. The HF eval path above needs none
> of this.

---

## 7. Knobs (all CLI-overridable — see `--help`)

| flag | default | notes |
|------|---------|-------|
| `--max-length` | 32768 | drop (not truncate) longer examples; tune from the dry-run p95 |
| `--lora-r` / `--lora-alpha` | 16 / 32 | rank 16–32 range; alpha = 2×r |
| `--lora-dropout` | 0.05 | |
| `--lora-target-modules` | q,k,v,o,gate,up,down `_proj` | Gemma attn+MLP; scoped to `language_model.*` via regex so the vision/audio towers aren't matched |
| `--learning-rate` | 2e-4 | LoRA-appropriate (higher than full-FT's ~2e-5) |
| `--num-train-epochs` | 3 | small corpus (2074 examples) — watch overfit; lower if train loss collapses |
| `--per-device-train-batch-size` | 1 | long sequences; raise only if VRAM allows |
| `--gradient-accumulation-steps` | 8 | the lever for effective batch |
| `--eval-frac` | 0.0 | hold out an SFT-loss eval set (distinct from the task eval in scripts/eval/eval.py) |
| `--no-assistant-only-loss` | off | train on the full sequence instead of assistant-only |
| `--report-to` | none | set `wandb`/`tensorboard` to log |

---

## 8. Design notes carried from CLAUDE.md (why the trainer looks the way it does)

- **No `device_map`.** `src/gemma/model.py`'s `load_model` hardcodes
  `device_map={"":0}` for single-GPU inference; that **breaks DDP** (accelerate
  raises "can't train a model loaded with device_map in distributed mode").
  `train_sft.py` uses its own loader that **reuses model.py's SDPA/GQA patch**
  (`_force_repeat_kv_for_efficient_sdpa`, which forces the mem-efficient kernel to
  serve Gemma 4's head_dim=512 global layers instead of OOMing on the MATH backend)
  but omits `device_map` and lets the Trainer place the model.
- **`attn_implementation="sdpa"`, never FlashAttention.** Gemma 4 E4B's global
  layers have `head_dim=512`; FA2/FA3 both cap head_dim at **256**, so they cannot
  run this model on any GPU — including the H200. Do **not** add `flash-attn`.
- **`packing=False`.** Padding-free/varlen packing requires FlashAttention (which we
  can't use), so packing under SDPA risks cross-document attention leakage. With only
  ~2k long examples, correctness beats throughput. (Future: packing becomes viable
  once document-masking under SDPA is verified.)
- **`assistant_only_loss` fallback.** Gemma 4's chat template has **no
  `{% generation %}` markers**, so TRL's `assistant_only_loss` is a **silent no-op**
  (it returns an all-zero assistant mask). Verified against the live tokenizer.
  `train_sft.py` instead computes the assistant-token mask itself from the template's
  **prefix-preservation** property and feeds TRL a **pre-tokenized** dataset
  (input_ids/attention_mask/labels); loss lands only on the model-generated tokens
  (each tool-call emission + the final JSON), with system/user/tool-declaration/
  tool-response bodies at -100. So `assistant_only_loss=False` in the SFTConfig **on
  purpose** — the masking is done upstream. This is on by default; disable with
  `--no-assistant-only-loss`.
- **`gradient_checkpointing=True`** (`use_reentrant=False`, `ddp_find_unused_
  parameters=False`): sequences are long, so activation memory matters more than the
  recompute tax even on H200.
- **Serve the merged bf16 checkpoint, never 4-bit base + adapter** (the v1
  QLoRA-fidelity finding: 4-bit serving of a bf16-trained adapter cost ~32 points of
  success rate — see [experiments.md](experiments.md)).
