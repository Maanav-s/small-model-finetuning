# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Fine-tune a small open-weight LLM (`google/gemma-4-E4B-it`) to take a restaurant name as input and return its menu as structured JSON, using **Firecrawl** (web search + scraping) as an inference-time tool. The full multi-phase plan — agentic tool-call loop → SFT distillation → GRPO RL → eval — lives in [project_plan.md](project_plan.md). Read it before working on any phase; it defines the target JSON schema and reward design.

> **Tooling note:** the plan switched from Tavily to **Firecrawl**, now wired in as an **MCP server** (`npx -y firecrawl-mcp`), not a direct API call — see "MCP integration" below. [project_plan.md](project_plan.md) still says Tavily in places (pending a docs pass). Two tool sources coexist: an offline `web_search` stub in [src/tools.py](src/tools.py) (returns [src/sample_menu.md](src/sample_menu.md); the deterministic default for dev) and the live Firecrawl MCP tools (enabled with `--mcp`).

**Status:** early scaffold (Phase 1). Code lives in [src/](src/), split by concern:

- [src/schema.py](src/schema.py) — the menu JSON **contract**: `SCHEMA_SNIPPET` (shown to the model), `MENU_SCHEMA` (machine-checkable), `extract_json`. The single source of truth the prompt, eval, and the future GRPO reward all import.
- [src/prompts.py](src/prompts.py) — `SYSTEM_PROMPT` / `MCP_SYSTEM_PROMPT`, built from `schema.SCHEMA_SNIPPET` so prompt and validator can't drift.
- [src/model.py](src/model.py) — `MODEL_ID` + `load_model(quantize, attn) -> (model, tokenizer)`.
- [src/tools.py](src/tools.py) — the `web_search` stub, `build_mcp_tools`, and `setup_tools(use_mcp)` (picks stub vs Firecrawl MCP, returns `(tools, registry, system_prompt, client)`).
- [src/mcp_client.py](src/mcp_client.py) — `MCPStdioClient`, a sync wrapper around the async MCP stdio SDK.
- [src/agent.py](src/agent.py) — **the agentic loop** (`build_messages`, `generate_turn`, `run_episode`). Takes `model`/`tokenizer`/`tools` as args with no import-time side effects, so you can drive it from a REPL/notebook (load once, re-run episodes as you edit prompts/tools). `uv run python src/agent.py` renders the prompt only (tokenizer-only, no GPU).
- [src/run_agent.py](src/run_agent.py) — thin CLI: `--quantize` (4-bit), `--attn sdpa|eager`, `--mcp` (Firecrawl). Loads `.env` for `FIRECRAWL_API_KEY`.

[main.py](main.py) forwards args to run_agent.py with `--quantize` forced on (the default on this host; see Hardware constraints) — e.g. `uv run python main.py --mcp`. Dev utilities live in [scripts/](scripts/) — e.g. [scripts/free_vram.sh](scripts/free_vram.sh) kills orphaned CUDA processes that pin VRAM after an interrupted run (`./scripts/free_vram.sh`, or `DRY_RUN=1` to list only).

## Environment & commands

- Package manager is **uv**. Run anything in the venv with `uv run python <script>.py` — do **not** call a bare `python`, and do not `pip install`.
- Add deps with `uv add <pkg>`; reproduce the env with `uv sync`. Commit `pyproject.toml` + `uv.lock`.
- The shell here is **bash** — use POSIX syntax for terminal commands (`$VAR`, `[ -f path ]`, `&&` to chain).

## Hardware constraints (these drive real code decisions)

Dev runs on an **Ubuntu instance with a 24 GB GPU** (e.g. RTX 3090/4090 / A10). With 24 GB the model is no longer VRAM-bound the way the old 6 GB laptop was, so some earlier constraints have relaxed:

- **`torch` is pinned to the CUDA `cu128` wheel index** via `[[tool.uv.index]]` + `[tool.uv.sources]` in [pyproject.toml](pyproject.toml). `explicit = true` means only torch uses it; everything else is PyPI. Drop to `cu126` only if the instance driver is unusually old.
- **bf16 fits in *VRAM* (~8–9 GB weights), but host RAM is the real bottleneck on this box: only ~15 GB total and no swap.** Loading bf16 materializes the full ~15 GB of weights in host RAM on the way to the GPU, which on a 15 GB box thrashes the page cache and makes the load crawl (re-reading the safetensors blob from disk) — *much* slower than the VRAM math suggests. 4-bit (`BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)`, ~3.5 GB) streams the weights down on load and sidesteps the cliff — that's why [main.py](main.py) defaults to `--quantize` here. [src/model.py](src/model.py) passes `low_cpu_mem_usage=True` for the bf16 path to cap peak host usage, but on a bigger-RAM instance bf16 is the better choice for full quality (and for leaving VRAM free for QLoRA training/RL batches, prefer 4-bit). If loads are still slow or OOM, run `./scripts/free_vram.sh` first — an interrupted run often leaves an orphaned python pinning the whole model in VRAM.
- **Use `device_map={"": 0}`, not `device_map="auto"`.** Pin to GPU 0 explicitly; `"auto"` can still offload across devices in ways you don't want. (On the old 6 GB card `"auto"` silently dumped the whole model to CPU — symptom: `Model loaded on: cpu`, `0.00 GB VRAM`, generation hangs.)
- **Attention backend: use `sdpa`, not FlashAttention-2.** FA2 is *incompatible with this model*: Gemma 4 E4B's global-attention layers use `head_dim=512` (`global_head_dim` in `config.json`; the sliding layers are 256), and FlashAttention-2 hard-caps head_dim at **256** on every GPU — so `attn_implementation="flash_attention_2"` loads but throws `FlashAttention forward only supports head dimension at most 256` at the first global layer. FA3/FA4 support larger head dims but require **Hopper** GPUs (H100); the 24 GB dev card is Ampere/Ada, so they don't apply either. Net: pass `attn_implementation="sdpa"` (torch built-in, no extra dep) — it handles 512-dim heads by falling back from its flash kernel to the mem-efficient/math path, and it's the default (`--attn sdpa|eager` on [src/run_agent.py](src/run_agent.py), applied in [src/model.py](src/model.py)'s `load_model`). Don't add a `flash-attn` dependency; it can't run this model. (Reaching for FA2 would also force a torch downgrade to ≤2.8, since no FA2 wheel is built for torch 2.11 and a source build OOMs this 15 GB / no-swap box — another reason to stay on `sdpa`.)

- **KV cache is 4-bit quantized** (`CACHE_IMPL="quantized"`, quanto backend, `nbits=4`, in [src/agent.py](src/agent.py)) to bound the VRAM that grows with sequence length as large scraped tool outputs accumulate in context. Independent of the weights' bnb quantization. Set `CACHE_IMPL=None` to fall back to the default cache. (Complementary: [src/tools.py](src/tools.py) also caps tool output *at the source* — see MCP integration.)

## Library version gotchas (transformers 5.x)

- `tokenizer.apply_chat_template(..., return_tensors="pt")` returns a **`BatchEncoding` dict**, not a bare tensor. Pass `return_dict=True` and call `model.generate(**inputs, ...)`; get the prompt length from `inputs["input_ids"].shape[-1]` to slice off the prompt when decoding.
- `apply_chat_template(..., tokenize=True)` **without** `return_dict` returns a **`tokenizers.Encoding`**, not a `list[int]` — read `.ids`, or pass `return_dict=True` and take `["input_ids"]`. (Easy to trip over when checking token-level prefix properties.)

## Gemma 4 chat template & tool calling

Gemma 4's template differs from Gemma 2/3 — verify behavior against the live tokenizer (render with `apply_chat_template(..., tokenize=False)`; cf. the `__main__` block in [src/agent.py](src/agent.py)) rather than assuming older-Gemma conventions. What we confirmed:

- **System role is supported natively** (`<|turn>system ... <turn|>`). No need to fold system text into the first user turn (that was a Gemma 2/3 limitation).
- **Define tools as plain Python functions** (typed signature + Google-style docstring) and pass them via `apply_chat_template(tools=[...])`. transformers auto-converts them to Gemma's schema; do **not** hand-write the tool-declaration string. System prompt and tool declarations coexist in the same system turn.
- **Wire format** (special tokens, not normal JSON — note `<|"|>` is the string delimiter and types are uppercase like `STRING`/`OBJECT`):
  - Declaration (system block): `<|tool>declaration:NAME{...}<tool|>`
  - Model emits a call: `<|tool_call>call:NAME{arg:<|"|>value<|"|>}<tool_call|>`
  - Tool result fed back: the loop appends the **bundled** form `{**parsed, "tool_responses": [{"name": NAME, "response": ...}]}` (the assistant turn carries both its `tool_calls` and their responses), which renders as `<|tool_response>response:NAME{value:<|"|>...<|"|>}<tool_response|>`. **A standalone `{"role": "tool", ...}` message renders only if it immediately follows an assistant turn with an *open* (unanswered) `tool_call`** — appended after a completed bundled turn (or as the first turn) it is **silently dropped (+0 tokens)**, and the response is always labeled by the preceding `tool_call`'s name (the message's own `name` is ignored). This is why the parse-failure recovery synthesizes an assistant tool-call turn rather than appending a bare tool message (see Prefix-preservation).
- **Parsing the model's output: decode with `skip_special_tokens=False`.** The `<|tool_call>` markers are special tokens — the default `skip_special_tokens=True` deletes them, so the loop would never see that a tool was called.

## Prefix-preservation & GRPO-readiness

Verified (tokenizer-only) that Gemma 4's template is **prefix-preserving for single-task agentic trajectories**: rendering `[m₀…mₖ]` is an exact token-prefix of `[m₀…mₖ, mₖ₊₁]` across `system → user → (assistant tool_call → tool response)* → final`, with or without `reasoning` traces. So **TRL's GRPO tool loop works on the stock template — no patched training template needed** (unlike Qwen3/DeepSeek-V3, which TRL auto-patches; Gemma 4 is not on that list).

The one way it breaks: the template's *reasoning guard* renders `reasoning`/`reasoning_content` only for assistant turns **after the last user message** (earlier ones are stripped via a `strip_thinking` macro). So a **mid-episode user turn rewrites earlier tokens** and breaks the prefix. Consequences:

- **Parse-failure recovery in [src/agent.py](src/agent.py).** On a `parse_response` failure (e.g. the model degenerates into a recursive/truncated `jsonOptions.schema` blob), `run_episode` feeds the error back as a **synthesized assistant tool-call turn carrying a tool response** — *not* a user turn (would break the prefix) and *not* a bare `{"role": "tool"}` message (silently dropped after a bundled turn → with greedy decoding the model just re-emits the same bad output). The recovery turn's tool name is regex-extracted from the malformed wire text (`call:NAME{…}`), falling back to the first registered tool.
- If you capture this loop's traces for SFT, they re-render consistently with how they'd appear in training.
- The thinking field the template reads is **`reasoning` / `reasoning_content`**, not `thinking`. `tokenizer.parse_response` may surface it under a different key — verify the round-trip if you need reasoning carried across tool turns in the inference loop.

## MCP integration (Firecrawl)

`--mcp` sources tools from the **Firecrawl MCP server** instead of the `web_search` stub. The pieces:

- **[src/mcp_client.py](src/mcp_client.py)** — `MCPStdioClient` runs one persistent asyncio loop on a background thread and keeps a single MCP session open (so the `npx` subprocess isn't respawned per call). The stdio + session contexts are entered *and* exited in the **same task** to avoid anyio's "cancel scope in a different task" error; `list_tools()` / `call_tool()` are sync wrappers over `run_coroutine_threadsafe`.
- **[src/tools.py](src/tools.py)** — `setup_tools(use_mcp=True)` launches `npx -y firecrawl-mcp` and calls `build_mcp_tools`, which lists the server's tools and converts each to the JSON-Schema dict `apply_chat_template(tools=...)` accepts directly (MCP tools have no Python signature, so this avoids synthesizing fake functions). Only `firecrawl_search` / `firecrawl_scrape` are **allowlisted** (Firecrawl exposes ~20 tools; the rest just bloat the prompt). `_apply_arg_policy` clamps args at dispatch — cap `firecrawl_search` `limit`, force `firecrawl_scrape` to compact markdown and never the `json`/`jsonOptions` extraction path — to bound context growth regardless of what the model emits.
- **`FIRECRAWL_API_KEY`** is required (the `--mcp` path errors without it). Loaded from a repo-root **`.env`** by [src/run_agent.py](src/run_agent.py) via `python-dotenv`. `.env` is git-ignored; commit [.env.example](.env.example) instead.
- **Requires Node / `npx` on PATH** (the server is a Node package). `mcp` and `python-dotenv` are project deps.

## Model access

`google/gemma-4-E4B-it` is **gated**. First load requires `huggingface-cli login` with a token from an account that has accepted the license on the model's HF page. Weights cache to `~/.cache/huggingface/hub`.
