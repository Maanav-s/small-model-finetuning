# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Fine-tune a small open-weight LLM (`google/gemma-4-E4B-it`) to take a restaurant name as input and return its menu as structured JSON, using **web search + scraping** as inference-time tools. The full multi-phase plan — agentic tool-call loop → SFT distillation → GRPO RL → eval — lives in [project_plan.md](project_plan.md). Read it before working on any phase; it defines the target JSON schema and reward design.

> **Tooling note:** the two live web tools are **finalized to Brave (search) + Jina (scrape)**, each called **directly via its REST API** (the earlier Tavily/Firecrawl/MCP-server wiring has been removed) — see "Web tools" below. [project_plan.md](project_plan.md) still says Tavily in places (pending a docs pass). Two tool sources coexist: the **live web tools** (`web_search` + `scrape_url`, the **default**) and an offline `web_search` stub in [src/tools.py](src/tools.py) (returns [src/sample_menu.md](src/sample_menu.md); a deterministic dev fallback, selected with `--offline`).

**Status:** early scaffold (Phase 1). Code lives in [src/](src/), split into **shared resources** (directly in `src/`) and two **per-model agent folders** ([src/gemma/](src/gemma/), [src/claude/](src/claude/)).

**Shared (in `src/`)** — model-agnostic, imported by both agents:

- [src/schema.py](src/schema.py) — the menu JSON **contract**: `SCHEMA_SNIPPET` (shown to the model), `MENU_SCHEMA` (machine-checkable), `extract_json`. The single source of truth the prompt, eval, and the future GRPO reward all import.
- [src/prompts.py](src/prompts.py) — `SYSTEM_PROMPT` / `LIVE_SYSTEM_PROMPT`, built from `schema.SCHEMA_SNIPPET` so prompt and validator can't drift; plus `TEST_RESTAURANT`, the single restaurant name both runners use as the episode input (temporary, for testing — real eval will iterate a dataset).
- [src/tools.py](src/tools.py) — the `web_search` stub, `build_model_tools`, and `setup_tools(offline=False)` (picks the live web tools vs stub, returns `(tools, registry, system_prompt)`). The live tools' backends live in [src/backends.py](src/backends.py) (`build_search` → Brave, `build_scrape` → Jina).

**Gemma agent (in `src/gemma/`)** — the local fine-tuning target:

- [src/gemma/model.py](src/gemma/model.py) — `MODEL_ID` + `load_model(quantize, attn) -> (model, tokenizer)`.
- [src/gemma/agent.py](src/gemma/agent.py) — **the agentic loop** (`build_messages`, `generate_turn`, `run_episode`). Takes `model`/`tokenizer`/`tools` as args with no import-time side effects, so you can drive it from a REPL/notebook (load once, re-run episodes as you edit prompts/tools). `uv run python src/gemma/agent.py` renders the prompt only (tokenizer-only, no GPU).
- [src/gemma/run_agent.py](src/gemma/run_agent.py) — thin CLI: `--quantize` (4-bit), `--attn sdpa|eager`, `--offline` (stub instead of the live web tools). Loads `.env` for `BRAVE_API_KEY` / `JINA_API_KEY`.

**Claude baseline (in `src/claude/`)** — a comparison point on the *same* tools/prompts/schema. See "Claude baseline" below.

- [src/claude/claude_agent.py](src/claude/claude_agent.py) — the loop driven through the Anthropic API (Claude Sonnet).
- [src/claude/run_claude.py](src/claude/run_claude.py) — thin CLI: `--offline`, `--model`. Loads `.env` for `ANTHROPIC_API_KEY`.

**Imports across the split:** the agent folders use flat imports (`from schema import ...`) and are run as scripts, so each entry module prepends the shared `src/` dir to `sys.path` (`sys.path.insert(0, str(Path(__file__).resolve().parent.parent))`) before importing shared modules — that's why those imports carry `# noqa: E402`. There are no `__init__.py` packages; keep the script-run convention.

[main.py](main.py) forwards args to [src/gemma/run_agent.py](src/gemma/run_agent.py) with `--quantize` forced on (the default on this host; see Hardware constraints) — e.g. `uv run python main.py` (live web tools) or `uv run python main.py --offline`. Dev utilities live in [scripts/](scripts/) — e.g. [scripts/free_vram.sh](scripts/free_vram.sh) kills orphaned CUDA processes that pin VRAM after an interrupted run (`./scripts/free_vram.sh`, or `DRY_RUN=1` to list only).

## Environment & commands

- Package manager is **uv**. Run anything in the venv with `uv run python <script>.py` — do **not** call a bare `python`, and do not `pip install`.
- Add deps with `uv add <pkg>`; reproduce the env with `uv sync`. Commit `pyproject.toml` + `uv.lock`.
- The shell here is **bash** — use POSIX syntax for terminal commands (`$VAR`, `[ -f path ]`, `&&` to chain).

## Hardware constraints (these drive real code decisions)

Dev runs on an **Ubuntu instance with a 24 GB GPU** (e.g. RTX 3090/4090 / A10). With 24 GB the model is no longer VRAM-bound the way the old 6 GB laptop was, so some earlier constraints have relaxed:

- **`torch` is pinned to the CUDA `cu128` wheel index** via `[[tool.uv.index]]` + `[tool.uv.sources]` in [pyproject.toml](pyproject.toml). `explicit = true` means only torch uses it; everything else is PyPI. Drop to `cu126` only if the instance driver is unusually old.
- **bf16 fits in *VRAM* (~8–9 GB weights), but host RAM is the real bottleneck on this box: only ~15 GB total and no swap.** Loading bf16 materializes the full ~15 GB of weights in host RAM on the way to the GPU, which on a 15 GB box thrashes the page cache and makes the load crawl (re-reading the safetensors blob from disk) — *much* slower than the VRAM math suggests. 4-bit (`BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)`, ~3.5 GB) streams the weights down on load and sidesteps the cliff — that's why [main.py](main.py) defaults to `--quantize` here. [src/gemma/model.py](src/gemma/model.py) passes `low_cpu_mem_usage=True` for the bf16 path to cap peak host usage, but on a bigger-RAM instance bf16 is the better choice for full quality (and for leaving VRAM free for QLoRA training/RL batches, prefer 4-bit). If loads are still slow or OOM, run `./scripts/free_vram.sh` first — an interrupted run often leaves an orphaned python pinning the whole model in VRAM.
- **Use `device_map={"": 0}`, not `device_map="auto"`.** Pin to GPU 0 explicitly; `"auto"` can still offload across devices in ways you don't want. (On the old 6 GB card `"auto"` silently dumped the whole model to CPU — symptom: `Model loaded on: cpu`, `0.00 GB VRAM`, generation hangs.)
- **Attention backend: use `sdpa`, not FlashAttention-2.** FA2 is *incompatible with this model*: Gemma 4 E4B's global-attention layers use `head_dim=512` (`global_head_dim` in `config.json`; the sliding layers are 256), and FlashAttention-2 hard-caps head_dim at **256** on every GPU — so `attn_implementation="flash_attention_2"` loads but throws `FlashAttention forward only supports head dimension at most 256` at the first global layer. FA3/FA4 support larger head dims but require **Hopper** GPUs (H100); the 24 GB dev card is Ampere/Ada, so they don't apply either. Net: pass `attn_implementation="sdpa"` (torch built-in, no extra dep) — it handles 512-dim heads by falling back from its flash kernel to the mem-efficient/math path, and it's the default (`--attn sdpa|eager` on [src/gemma/run_agent.py](src/gemma/run_agent.py), applied in [src/gemma/model.py](src/gemma/model.py)'s `load_model`). Don't add a `flash-attn` dependency; it can't run this model. (Reaching for FA2 would also force a torch downgrade to ≤2.8, since no FA2 wheel is built for torch 2.11 and a source build OOMs this 15 GB / no-swap box — another reason to stay on `sdpa`.)
- **`sdpa` needs a GQA patch or it OOMs on long context.** With plain `sdpa`, Gemma 4's global layers (`head_dim=512`, GQA: 8 query / 2 KV heads) hit a transformers heuristic (`use_gqa_in_sdpa`) that, for mask-free causal layers, keeps KV un-expanded and passes `enable_gqa=True`. PyTorch's mem-efficient kernel **can't do the GQA broadcast at head_dim=512** on Ampere/Ada, so SDPA silently falls to the **MATH** backend and materializes the `(1, 8, S, S)` score matrix — measured OOM at ~14k tokens (baseline 13 GB at 6k, then OOM). [src/gemma/model.py](src/gemma/model.py)'s `load_model` fixes this by forcing the `repeat_kv` path (`use_gqa_in_sdpa → False`): with matched head counts the efficient kernel serves head_dim=512, and **default SDPA dispatch picks it on its own — no `sdpa_kernel(...)` override** (an explicit one was a no-op; the GQA broadcast, not kernel priority, was the disqualifier). Result: prefill stays **linear** in S (12.2 GB @ 14k, 13.5 GB @ 20k) instead of OOMing. `flex_attention` is *not* an alternative here — its Triton kernel exceeds Ampere/Ada shared memory at head_dim=512.

## Library version gotchas (transformers 5.x)

- `tokenizer.apply_chat_template(..., return_tensors="pt")` returns a **`BatchEncoding` dict**, not a bare tensor. Pass `return_dict=True` and call `model.generate(**inputs, ...)`; get the prompt length from `inputs["input_ids"].shape[-1]` to slice off the prompt when decoding.
- `apply_chat_template(..., tokenize=True)` **without** `return_dict` returns a **`tokenizers.Encoding`**, not a `list[int]` — read `.ids`, or pass `return_dict=True` and take `["input_ids"]`. (Easy to trip over when checking token-level prefix properties.)

## Gemma 4 chat template & tool calling

Gemma 4's template differs from Gemma 2/3 — verify behavior against the live tokenizer (render with `apply_chat_template(..., tokenize=False)`; cf. the `__main__` block in [src/gemma/agent.py](src/gemma/agent.py)) rather than assuming older-Gemma conventions. What we confirmed:

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

- **Parse-failure recovery in [src/gemma/agent.py](src/gemma/agent.py).** On a `parse_response` failure (e.g. the model degenerates into a recursive/truncated `jsonOptions.schema` blob), `run_episode` feeds the error back as a **synthesized assistant tool-call turn carrying a tool response** — *not* a user turn (would break the prefix) and *not* a bare `{"role": "tool"}` message (silently dropped after a bundled turn → with greedy decoding the model just re-emits the same bad output). The recovery turn's tool name is regex-extracted from the malformed wire text (`call:NAME{…}`), falling back to the first registered tool.
- If you capture this loop's traces for SFT, they re-render consistently with how they'd appear in training.
- The thinking field the template reads is **`reasoning` / `reasoning_content`**, not `thinking`. `tokenizer.parse_response` may surface it under a different key — verify the round-trip if you need reasoning carried across tool turns in the inference loop.

## Web tools (Brave + Jina, direct REST)

By default the agent sources two live tools — **`web_search` backed by Brave, `scrape_url` backed by Jina** — each called **directly via its REST API** (no SDK, no MCP server, no `npx` subprocess); `--offline` swaps in the `web_search` stub. The pieces:

- **[src/backends.py](src/backends.py)** — the network seam. `build_search()` returns a Brave-backed `search(query) -> str` (`GET /res/v1/web/search`, `X-Subscription-Token` auth); `build_scrape()` returns a Jina-backed `scrape(url) -> str` (`GET r.jina.ai/<url>`, `Authorization: Bearer`, `X-Return-Format: markdown`). `search` pins `count=SEARCH_RESULT_LIMIT` and renders results through one shared `_format_results` so the model sees identical search output; `scrape` returns clean markdown. These functions return **un-capped** strings.
- **[src/tools.py](src/tools.py)** — `setup_tools(offline=False)` calls `build_search()`/`build_scrape()` and wraps them via `build_model_tools` into two **plain Python functions** (`web_search`/`scrape_url`) with fixed, vendor-neutral docstrings — the model never sees a provider name (important: the tool names get baked into the SFT/GRPO training data). They're ordinary callables with typed signatures + docstrings, so `apply_chat_template(tools=...)` and the Claude runner's `to_anthropic_tools` consume them exactly like the stub. The wrappers add the `MAX_TOOL_CHARS` cap (with a non-silent truncation warning) and nothing else. **Caching is not implemented yet**; when added it wraps the two backend calls (disk cache keyed on normalized args, gitignored) and nothing else changes.
- **`BRAVE_API_KEY` + `JINA_API_KEY`** are required for the live (default) path — `build_search`/`build_scrape` raise a `SystemExit` pointing at `--offline` if their key is missing. Loaded from a repo-root **`.env`** by [src/gemma/run_agent.py](src/gemma/run_agent.py) via `python-dotenv`. `.env` is git-ignored; commit [.env.example](.env.example) instead.
- **No Node/`npx` and no vendor SDK** — both tools are plain `requests` calls. `requests` and `python-dotenv` are project deps; the old `mcp`/`firecrawl-py`/`markdownify` deps and `src/mcp_client.py` were removed.

## Claude baseline (Anthropic API)

A second runner drives **Claude Sonnet** (`claude-sonnet-4-6`, in [src/claude/claude_agent.py](src/claude/claude_agent.py)) through the same task, so we can compare a frontier model against Gemma on identical inputs. The point is parity: it reuses the *same* tool source, system prompt, and JSON contract — only the model and the transport differ.

- **Shared everything.** [src/claude/run_claude.py](src/claude/run_claude.py) calls the same `setup_tools(offline=...)` from [src/tools.py](src/tools.py) (live Brave+Jina tools or stub `web_search`) and the same `SYSTEM_PROMPT`/`LIVE_SYSTEM_PROMPT`, and validates with the same `schema.extract_json`. The tool *registry* (`name -> callable -> str`) is used as-is; only the tool *declaration* format is translated.
- **Tool translation.** Both tool sources are now plain Python callables (typed signature + docstring); the Anthropic API wants `{"name","description","input_schema"}`. `to_anthropic_tools` in [src/claude/claude_agent.py](src/claude/claude_agent.py) derives that from each callable's signature/docstring (the old OpenAI-style function-dict branch is gone with MCP), so the same `setup_tools` result drives Claude unchanged. `uv run python src/claude/claude_agent.py` prints the converted declarations (no key/network needed).
- **The loop** is a standard manual agentic loop over `client.messages.create` (no transformers, no GPU): call → run `tool_use` blocks via the registry → feed `tool_result`s back → repeat. Adaptive thinking is on (`thinking={"type":"adaptive"}`); the full assistant turn (thinking + tool_use blocks) is appended verbatim each round, as the API requires. Same budget as the Gemma loop (`MAX_TOOL_CALLS=4`, `MAX_TOKENS=4096`) — but on budget-exhaustion it makes one **tool-free** call to force a JSON answer (Gemma's loop returns `""` instead).
- **Keys & deps.** Requires `ANTHROPIC_API_KEY` (repo-root `.env`, loaded by `python-dotenv`; the runner errors before any network call if it's missing). The live (default) tool path additionally needs `BRAVE_API_KEY` + `JINA_API_KEY`, exactly like the Gemma path. The `anthropic` SDK is a project dep. Run: `uv run python src/claude/run_claude.py` (live web tools) or `--offline` (stub).

## Model access

`google/gemma-4-E4B-it` is **gated**. First load requires `huggingface-cli login` with a token from an account that has accepted the license on the model's HF page. Weights cache to `~/.cache/huggingface/hub`.
