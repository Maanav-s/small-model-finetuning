# RunPod SFT run — Gemma 4 E4B menu extractor (2× H200)

Setup + run guide for the LoRA SFT stage ([scripts/train_sft.py](../scripts/train_sft.py))
on a **RunPod pod with 2× NVIDIA H200 (141 GB each, Hopper / sm_90)**. This is the
distillation step: the student learns the teacher's trajectories re-rendered under
the **student** prompt (context distillation — see CLAUDE.md). Data comes from
`data/sft/train.jsonl` ([scripts/build_sft.py](../scripts/build_sft.py)).

Unlike the dev box (1× 24 GB, 15 GB host RAM, no swap), the H200 pod is not
memory-bound: we load the base in **bf16** (not 4-bit) and LoRA-train it. A 4B +
LoRA fits in one H200 many times over, so we use plain **DDP** (one process per
GPU, gradients all-reduced) — no DeepSpeed/FSDP.

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
git clone <this-repo> small-model-finetuning   # or rsync it up
cd small-model-finetuning
```

Bring the SFT dataset onto the pod (it is git-ignored; source of truth is S3):

```bash
# option A: pull the whole data/ snapshot from S3 (needs AWS creds on the pod)
uv run python scripts/cache_sync.py pull        # brings data/ incl. traces + cache
uv run python scripts/build_sft.py              # traces/ -> data/sft/train.jsonl
# option B: just scp data/sft/train.jsonl up if you built it elsewhere
```

`train_sft.py` only needs `data/sft/train.jsonl`. It does **no** scraping, so you do
**not** need `playwright install` on the pod.

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
uv run python scripts/train_sft.py --dry-run --data data/sft/train.jsonl
```

Read the printed **length p50/p90/p95/p99/max**. Examples longer than `--max-length`
are **dropped** (never right-truncated — that would cut off the final-JSON target).

> **Measured on the current data (131 rows, `MAX_TOOL_CHARS=75000`):** episodes are
> long — **p50 ≈ 14k, p90 ≈ 61k, p95 ≈ 95k, max ≈ 209k tokens** (several capped
> scraped pages per episode). Keep-rate by `--max-length`: 16384 → 57%, **32768 →
> 80% (the default)**, 49152 → 85%, 65536 → 92%, 98304 → 95%.
>
> The default `--max-length 32768` deliberately does **not** chase p95: a ~98k-token
> sequence's LM-head logits (`seq × vocab × 2 bytes` ≈ 50 GB, doubled for grads)
> risk OOM even on a 141 GB H200. **The right lever for the long tail is a lower
> `MAX_TOOL_CHARS` in `build_sft.py`, then rebuild** — that shrinks every episode at
> the source instead of dropping whole trajectories. Raise `--max-length` past 32768
> only after verifying memory headroom (watch the first steps). The >10% drop
> warning is expected at the default with this data; it's the signal to lower the
> tool-result cap upstream.

---

## 5. Launch training (under tmux)

tmux survives an SSH drop (it does **not** survive a pod restart — that's why
checkpoints go to `/workspace`).

```bash
tmux new -s sft
cd /workspace/small-model-finetuning

accelerate launch --config_file configs/accelerate_ddp.yaml \
    scripts/train_sft.py \
    --data data/sft/train.jsonl \
    --output-dir /workspace/gemma-menu-sft \
    --max-length 16384 \
    --num-train-epochs 3 \
    --lora-r 16 --lora-alpha 32 \
    --learning-rate 2e-4
```

Detach with `Ctrl-b d`; re-attach with `tmux attach -t sft`.

**torchrun equivalent** (same result, if you prefer it over accelerate):

```bash
torchrun --nproc_per_node=2 scripts/train_sft.py \
    --data data/sft/train.jsonl --output-dir /workspace/gemma-menu-sft
```

> **You MUST use accelerate/torchrun.** Plain `python scripts/train_sft.py` launches
> a **single** process — it trains on GPU 0 only and leaves GPU 1 idle (no DDP).

### Effective batch size

`effective batch = per_device_train_batch_size × gradient_accumulation_steps × num_GPUs`.
With the defaults: **1 × 8 × 2 = 16**. The trainer reads `WORLD_SIZE` (set by
accelerate/torchrun) for the GPU count, so the printed math is correct under the
launcher. To change the effective batch, adjust `--gradient-accumulation-steps`
(memory-free) rather than `--per-device-train-batch-size` (H200 has headroom, but
long sequences dominate activation memory even with gradient checkpointing on).

---

## 6. Outputs

Written under `--output-dir` (put it on `/workspace`):

- `…/adapter/` — the LoRA adapter (`save_pretrained`) + tokenizer.
- `…/merged/`  — the adapter merged into the bf16 base (`merge_and_unload`) +
  tokenizer, a **plain HF model** so the eval runner needs no peft:

  ```bash
  uv run python scripts/eval_split.py --model gemma --model-path /workspace/gemma-menu-sft/merged
  ```

The merge runs on the main process only (after `wait_for_everyone`), reloading the
base to keep it DDP-wrapper-free. Push the outputs off the pod before terminating
(they're only on `/workspace`, which survives termination, but back them up / sync
to S3 to be safe).

---

## 7. Knobs (all CLI-overridable — see `--help`)

| flag | default | notes |
|------|---------|-------|
| `--max-length` | 16384 | drop (not truncate) longer examples; tune from the dry-run p95 |
| `--lora-r` / `--lora-alpha` | 16 / 32 | rank 16–32 range; alpha = 2×r |
| `--lora-dropout` | 0.05 | |
| `--lora-target-modules` | q,k,v,o,gate,up,down `_proj` | Gemma attn+MLP; matched by suffix |
| `--learning-rate` | 2e-4 | LoRA-appropriate (higher than full-FT's ~2e-5) |
| `--num-train-epochs` | 3 | small corpus — watch overfit; lower if train loss collapses |
| `--per-device-train-batch-size` | 1 | long sequences; raise only if VRAM allows |
| `--gradient-accumulation-steps` | 8 | the lever for effective batch |
| `--eval-frac` | 0.0 | hold out an SFT-loss eval set (distinct from the WS-G task eval) |
| `--no-assistant-only-loss` | off | train on the full sequence instead of assistant-only |
| `--report-to` | none | set `wandb`/`tensorboard` to log |

---

## 8. Design notes carried from CLAUDE.md (why the trainer looks the way it does)

- **No `device_map`.** `model.py`'s `load_model` hardcodes `device_map={"":0}` for
  single-GPU inference; that **breaks DDP** (accelerate raises "can't train a model
  loaded with device_map in distributed mode"). `train_sft.py` uses its own loader
  that **reuses model.py's SDPA/GQA patch** (`_force_repeat_kv_for_efficient_sdpa`,
  which forces the mem-efficient kernel to serve Gemma 4's head_dim=512 global
  layers instead of OOMing on the MATH backend) but omits `device_map` and lets the
  Trainer place the model.
- **`attn_implementation="sdpa"`, never FlashAttention.** Gemma 4 E4B's global
  layers have `head_dim=512`; FA2/FA3 both cap head_dim at **256**, so they cannot
  run this model on any GPU — including the H200. Do **not** add `flash-attn`.
- **`packing=False`.** Padding-free/varlen packing requires FlashAttention (which we
  can't use), so packing under SDPA risks cross-document attention leakage. With only
  ~1000 long examples, correctness beats throughput. (Future: packing becomes viable
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
