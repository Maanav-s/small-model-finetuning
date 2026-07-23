# small-model-finetuning

Fine-tune a small open-weight LLM (`google/gemma-4-E4B-it`) to take a **restaurant name** as input and return its **menu as structured JSON**, using **web search + scraping as inference-time tools**. The model doesn't memorize menus — it learns to *drive the tools* (search → scrape → extract) and emit the schema in [src/schema.py](src/schema.py).

For the detailed engineering notes and hard-won constraints (the chat template, attention backend, vLLM serving, etc.), read [CLAUDE.md](CLAUDE.md) — it is the living source of truth. Planning/spec docs live in [notes/](notes/).

## Pipeline

| Phase | Status |
|---|---|
| Agentic tool-call loop | ✅ done |
| Tool-call caching + teacher corpus | ✅ done — 2233 teacher traces (2074 kept after cleaning) |
| **SFT distillation** | 🔜 current |
| GRPO RL | planned |
| Eval | planned |

The **teacher** is a self-hosted `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8` on vLLM (4×H100); the **student** is Gemma-4-E4B-it, trained on the teacher's trajectories re-rendered under a leaner prompt (context distillation — see CLAUDE.md).

## Layout

```
src/                     shared, model-agnostic modules
  schema.py              the menu JSON contract (single source of truth)
  prompts.py             system prompts (teacher/student variants)
  tools.py, backends.py  web_search (Brave) + scrape_url (local Chromium)
  cache.py               content-addressed SQLite tool-call cache
  corpus.py              data/corpus.sqlite access (the single data source of truth)
  grounding.py           menu-grounding scorer
  episodes.py, run_meta.py, reward.py, eval_metrics.py
  gemma/                 the fine-tuning target: model.py, agent.py, run_agent.py
  claude/                Claude Sonnet baseline on the same tools/prompts
  serving/               openai_agent.py — vLLM/OpenAI-compatible runner
scripts/
  corpus/                harvest, build_corpus, warm_cache, clean_cache, assign_splits
  datasets/              build_sft, build_grpo (corpus → training jsonl)
  train/                 train_sft, train_grpo, to_text_only
  eval/                  eval.py
  analysis/              analyze_queries, analyze_tool_chars
  infra/                 corpus_sync (S3), runpod_create, serve_teacher.sh, smoke_teacher
data/                    git-ignored; source of truth is S3 (see notes/S3_setup.md)
  corpus.sqlite          restaurants + sft/grpo/eval split + traces + grounding + reject flags
  cache.sqlite           tool-call cache
  sft/<family>/train.jsonl   SFT export
notes/                   plans + experiment log
viz/                     local FastAPI trace-review / data-cleaning UI
```

## Setup

Package manager is **uv** (never bare `pip`). The shell examples assume bash.

```bash
uv sync                        # reproduce the env from pyproject.toml + uv.lock
uv run playwright install chromium   # scrape backend
cp .env.example .env           # then fill in BRAVE_API_KEY, HF_TOKEN, etc.
huggingface-cli login          # google/gemma-4-E4B-it is gated
```

`.env` holds `BRAVE_API_KEY` (search; scrape needs no key), `HF_TOKEN` (gated model), the Claude baseline's `ANTHROPIC_API_KEY`, and the S3 config (`S3_BUCKET`/`S3_PREFIX`/`AWS_*`). See [.env.example](.env.example).

## Running the stages

```bash
# Agentic loop on one episode (dev box defaults to 4-bit — see CLAUDE.md hardware notes)
uv run python main.py

# Claude baseline on the same task
uv run python src/claude/run_claude.py

# Sync data artifacts with S3 (corpus, cache, datasets, models)
uv run python scripts/infra/corpus_sync.py pull        # S3 -> local
uv run python scripts/infra/corpus_sync.py push --only sft

# Build the SFT dataset from the corpus, then LoRA-train (2×H200 pod — see notes/runpod_sft.md)
uv run python scripts/datasets/build_sft.py
uv run python scripts/train/train_sft.py --dry-run     # cheap gate before GPU
accelerate launch --config_file configs/accelerate_ddp.yaml scripts/train/train_sft.py

# Review / clean traces locally (writes the reject flags in corpus.sqlite)
uv run uvicorn viz.review:app --host 127.0.0.1 --port 8001
```

## Hardware note

The dev box is a single 24 GB GPU with limited host RAM, so `main.py` defaults to 4-bit loading and the code pins `sdpa` attention (FlashAttention cannot serve Gemma-4's `head_dim=512` layers). Real training/serving happens on rented multi-GPU pods. These constraints and their reasons are documented in detail in [CLAUDE.md](CLAUDE.md).
