# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Fine-tune a small open-weight LLM (`google/gemma-4-E4B-it`) to take a restaurant name as input and return its menu as structured JSON, using **Firecrawl** (web search + scraping) as an inference-time tool. The full multi-phase plan — agentic tool-call loop → SFT distillation → GRPO RL → eval — lives in [project_plan.md](project_plan.md). Read it before working on any phase; it defines the target JSON schema and reward design.

> **Tooling note:** the plan switched from Tavily to **Firecrawl**. [project_plan.md](project_plan.md) still says Tavily in places, and the `web_search` tool in [src/agent.py](src/agent.py) still has a Tavily call body — both are pending migration to Firecrawl. The *tool interface* (a `web_search(query)` Python function passed to the chat template) stays the same; only the implementation inside it changes.

**Status:** early scaffold. Code lives in [src/](src/): [src/agent.py](src/agent.py) defines the system prompt + tool schema (Phase 1), and [src/run_agent.py](src/run_agent.py) drives the generate/execute tool-call loop (run with `uv run python src/run_agent.py`; pass `--quantize` for 4-bit on a small card). `main.py` in the root is an unused stub. Dev utilities live in [scripts/](scripts/) — e.g. [scripts/free_vram.sh](scripts/free_vram.sh) kills orphaned CUDA processes that pin VRAM after an interrupted run.

## Environment & commands

- Package manager is **uv**. Run anything in the venv with `uv run python <script>.py` — do **not** call a bare `python`, and do not `pip install`.
- Add deps with `uv add <pkg>`; on the Linux cluster, reproduce the env with `uv sync`. Commit `pyproject.toml` + `uv.lock`.
- The shell here is **PowerShell**, not bash — use PowerShell syntax for terminal commands (`$env:VAR`, `Test-Path`, `;` to chain).

## Hardware constraints (these drive real code decisions)

Local dev is a Windows laptop with an **RTX 4050, 6 GB VRAM**. This is the binding constraint for anything that loads the model:

- **`torch` is pinned to the CUDA `cu128` wheel index** via `[[tool.uv.index]]` + `[tool.uv.sources]` in [pyproject.toml](pyproject.toml). `explicit = true` means only torch uses it; everything else is PyPI. The same config produces Linux CUDA wheels on the cluster — no changes needed unless the cluster driver is unusually old (then drop to `cu126`).
- **The model will not fit in bf16** (~8–9 GB weights). Load it in **4-bit** via `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)` (~3.5 GB). bf16 is the right compute dtype — the GPU supports it.
- **Use `device_map={"": 0}`, not `device_map="auto"`.** On a 6 GB card `"auto"` silently offloads the whole model to CPU (symptom: `Model loaded on: cpu`, `0.00 GB VRAM`, generation hangs for minutes). Pin to GPU 0 explicitly.

## Library version gotchas (transformers 5.x)

- `tokenizer.apply_chat_template(..., return_tensors="pt")` returns a **`BatchEncoding` dict**, not a bare tensor. Pass `return_dict=True` and call `model.generate(**inputs, ...)`; get the prompt length from `inputs["input_ids"].shape[-1]` to slice off the prompt when decoding.

## Gemma 4 chat template & tool calling

Gemma 4's template differs from Gemma 2/3 — verify behavior against the live tokenizer (render with `apply_chat_template(..., tokenize=False)`; cf. the `__main__` block in [src/agent.py](src/agent.py)) rather than assuming older-Gemma conventions. What we confirmed:

- **System role is supported natively** (`<|turn>system ... <turn|>`). No need to fold system text into the first user turn (that was a Gemma 2/3 limitation).
- **Define tools as plain Python functions** (typed signature + Google-style docstring) and pass them via `apply_chat_template(tools=[...])`. transformers auto-converts them to Gemma's schema; do **not** hand-write the tool-declaration string. System prompt and tool declarations coexist in the same system turn.
- **Wire format** (special tokens, not normal JSON — note `<|"|>` is the string delimiter and types are uppercase like `STRING`/`OBJECT`):
  - Declaration (system block): `<|tool>declaration:NAME{...}<tool|>`
  - Model emits a call: `<|tool_call>call:NAME{arg:<|"|>value<|"|>}<tool_call|>`
  - Tool result fed back: pass `{"role": "tool", "name": NAME, "content": ...}`, which renders as `<|tool_response>response:NAME{value:<|"|>...<|"|>}<tool_response|>`
- **Parsing the model's output: decode with `skip_special_tokens=False`.** The `<|tool_call>` markers are special tokens — the default `skip_special_tokens=True` deletes them, so the loop would never see that a tool was called.

## Model access

`google/gemma-4-E4B-it` is **gated**. First load requires `huggingface-cli login` with a token from an account that has accepted the license on the model's HF page. Weights cache to `C:\Users\<user>\.cache\huggingface\hub` (outside the OneDrive-synced project dir — keep it that way). On Windows the HF symlink warning is harmless (no Developer Mode → files are copied, not symlinked).
