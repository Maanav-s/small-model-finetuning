# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Fine-tune a small open-weight LLM (`google/gemma-4-E4B-it`) to take a restaurant name as input and return its menu as structured JSON, using **Firecrawl** (web search + scraping) as an inference-time tool. The full multi-phase plan — agentic tool-call loop → SFT distillation → GRPO RL → eval — lives in [project_plan.md](project_plan.md). Read it before working on any phase; it defines the target JSON schema and reward design.

> **Tooling note:** the plan switched from Tavily to **Firecrawl**. [project_plan.md](project_plan.md) still says Tavily in places, and the `web_search` tool in [src/agent.py](src/agent.py) still has a Tavily call body — both are pending migration to Firecrawl. The *tool interface* (a `web_search(query)` Python function passed to the chat template) stays the same; only the implementation inside it changes.

**Status:** early scaffold. Code lives in [src/](src/): [src/agent.py](src/agent.py) defines the system prompt + tool schema (Phase 1), and [src/run_agent.py](src/run_agent.py) drives the generate/execute tool-call loop (run with `uv run python src/run_agent.py`; pass `--quantize` for 4-bit on a small card). `main.py` in the root is an unused stub. Dev utilities live in [scripts/](scripts/) — e.g. [scripts/free_vram.sh](scripts/free_vram.sh) kills orphaned CUDA processes that pin VRAM after an interrupted run.

## Environment & commands

- Package manager is **uv**. Run anything in the venv with `uv run python <script>.py` — do **not** call a bare `python`, and do not `pip install`.
- Add deps with `uv add <pkg>`; reproduce the env with `uv sync`. Commit `pyproject.toml` + `uv.lock`.
- The shell here is **bash** — use POSIX syntax for terminal commands (`$VAR`, `[ -f path ]`, `&&` to chain).

## Hardware constraints (these drive real code decisions)

Dev runs on an **Ubuntu instance with a 24 GB GPU** (e.g. RTX 3090/4090 / A10). With 24 GB the model is no longer VRAM-bound the way the old 6 GB laptop was, so some earlier constraints have relaxed:

- **`torch` is pinned to the CUDA `cu128` wheel index** via `[[tool.uv.index]]` + `[tool.uv.sources]` in [pyproject.toml](pyproject.toml). `explicit = true` means only torch uses it; everything else is PyPI. Drop to `cu126` only if the instance driver is unusually old.
- **bf16 now fits** (~8–9 GB weights), so 4-bit is **optional, not required**. For plain inference, load in bf16 (`torch_dtype=torch.bfloat16`) for full quality with headroom to spare. Keep **4-bit** (`BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)`, ~3.5 GB) when you want to leave room for training/RL batches and optimizer states (QLoRA). `test.py` still loads 4-bit — fine as a smoke test, not a mandate.
- **Use `device_map={"": 0}`, not `device_map="auto"`.** Pin to GPU 0 explicitly; `"auto"` can still offload across devices in ways you don't want. (On the old 6 GB card `"auto"` silently dumped the whole model to CPU — symptom: `Model loaded on: cpu`, `0.00 GB VRAM`, generation hangs.)
- **Flash Attention is now worth it.** On Linux `flash-attn` installs cleanly (no Windows build pain), and the 24 GB card supports FA2 — useful for the long prefills this project produces when scraped Firecrawl content is fed back as tool results. Pass `attn_implementation="flash_attention_2"` to `from_pretrained`; `"sdpa"` (torch built-in, no extra dep) is the zero-install fallback. Decode is still weight-bandwidth-bound, so the gains are mostly on prefill + memory.

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

`google/gemma-4-E4B-it` is **gated**. First load requires `huggingface-cli login` with a token from an account that has accepted the license on the model's HF page. Weights cache to `~/.cache/huggingface/hub`.
