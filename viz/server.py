"""Local visualizer for the Gemma menu-extraction agent.

A single FastAPI process that:
  - loads the Gemma model + an Anthropic client ONCE at startup (in-process),
  - serves the static page (static/index.html) at "/", and
  - exposes POST /api/extract {"query", "agent", "dietary", "prompt_variant"}
    -> menu JSON.

Either agent runs the SAME tools / system prompt / JSON contract (only the loop
differs): the local Gemma model (gemma/agent.py) or Claude via the Anthropic API
(claude/claude_agent.py). The default is gemma.

Search is Brave; scrape is a local headless Chromium (requests fast-path +
auto-scrolling pooled browser -- see src/backends.py). The scrape's sync
Playwright browser is safe here because /api/extract is a sync def: FastAPI runs
it in a threadpool worker with no event loop. Do NOT make it `async def` without
moving the scrape onto its own thread. Tools are built lazily and cached once.

Episodes are serialized behind one lock: one runs at a time no matter how many
browser tabs hit it. For Gemma this is essential (concurrent generate() calls on
the single GPU would race / OOM). FastAPI runs the sync endpoint in a threadpool,
so the lock -- not the event loop -- does the gating.

Run from the repo root:
    uv run uvicorn viz.server:app --host 127.0.0.1 --port 8000

Set BRAVE_API_KEY in the repo-root .env for the live tools (scrape runs locally,
no key). The Claude agent additionally needs ANTHROPIC_API_KEY; without it, only
Gemma is offered (a Claude request returns an error rather than failing at startup).
"""

from __future__ import annotations

import os
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

# The src/ modules use flat imports (`from model import ...`, `from agent import
# ...`) and expect src/ and the per-agent folders on sys.path -- mirror the
# convention run_agent.py / run_claude.py set up so we can reuse both engines.
_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
for _p in (_SRC, _SRC / "gemma", _SRC / "claude"):
    sys.path.insert(0, str(_p))

load_dotenv(_REPO / ".env")  # BRAVE_API_KEY (+ ANTHROPIC_API_KEY for Claude)

# Imported after sys.path is set. model.py sets PYTORCH_CUDA_ALLOC_CONF before it
# touches torch, so it must be the first of these to import. The two run_episode
# loops share a name, so alias them.
from model import load_model                          # noqa: E402
from backends import has_search_key                   # noqa: E402
from tools import setup_tools                          # noqa: E402
from prompts import build_system_prompt                # noqa: E402
from agent import run_episode as run_gemma_episode     # noqa: E402
from claude_agent import (                              # noqa: E402
    HAIKU_MODEL_ID,
    MODEL_ID as CLAUDE_SONNET_ID,
    run_episode as run_claude_episode,
)
from schema import extract_json                        # noqa: E402

# UI agent value -> Claude model id. Both run the SAME claude_agent loop; only the
# model differs (and the loop picks the right thinking config per model).
CLAUDE_AGENTS = {"claude": CLAUDE_SONNET_ID, "claude-haiku": HAIKU_MODEL_ID}

# 4-bit by default on this box (15 GB host RAM; see CLAUDE.md). Set VIZ_QUANTIZE=0
# on a bigger-RAM machine to load full-quality bf16 instead.
_QUANTIZE = os.environ.get("VIZ_QUANTIZE", "1") != "0"

# VIZ_ADAPTER: path to a trained PEFT adapter (e.g. models/sft-adapter) to load on
# top of the base. Unset = the raw base model. This is how the viz serves the
# SHIPPED student rather than an untrained Gemma.
_ADAPTER = os.environ.get("VIZ_ADAPTER", "").strip()
# A label for the UI, so a screenshot can never be mistaken for the wrong model.
_CHECKPOINT = (Path(_ADAPTER).name if _ADAPTER else "base (untrained)")

_ENGINE: dict = {}                 # model/tokenizer/client, populated at startup
_EPISODE_LOCK = threading.Lock()   # serialize episodes (single GPU: concurrent generate() would race/OOM)

# The live tools (Brave search + local Chromium scrape) are built lazily and
# cached here, so the server boots with no web key and the key is only needed the
# first time an extraction runs. _TOOLS_LOCK guards the cache against concurrent
# first builds. Only (tools, registry) are cached -- the system prompt varies per
# request with the caller's dietary restrictions, so it's rebuilt each episode.
_TOOLS_CACHE: list = []
_TOOLS_LOCK = threading.Lock()


def _get_tools() -> tuple:
    """Return the dietary-independent (tools, registry), building+caching once.

    scrape_url is backed by a local headless Chromium (src/backends.py) that uses
    the SYNC Playwright API. That is safe here ONLY because the /api/extract
    endpoint is a sync def -- FastAPI runs it in a threadpool worker (no running
    event loop in that thread), so the browser launch is effectively
    thread-offloaded. Do NOT make the endpoint `async def` without moving the
    scrape onto its own thread first, or sync Playwright will raise.

    Raises SystemExit (from setup_tools) if the search key is missing; the caller
    turns that into a clean API error instead of crashing the server.
    """
    with _TOOLS_LOCK:
        if not _TOOLS_CACHE:
            tools, registry, _ = setup_tools()
            _TOOLS_CACHE.append((tools, registry))
        return _TOOLS_CACHE[0]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model and Anthropic client once at startup (tools are built lazily)."""
    model, tokenizer = load_model(quantize=_QUANTIZE, attn="sdpa")
    if _ADAPTER:
        # REFUSE 4-bit + adapter. The adapter was trained against a bf16 base, and
        # serving it on a 4-bit one is the single most expensive mistake in this
        # project's history: it cost ~32 points of success rate (24% -> 56% when
        # fixed; see notes/experiments.md). A demo that silently does this shows
        # numbers far below the README's and looks like the model is bad.
        if _QUANTIZE:
            raise SystemExit(
                "VIZ_ADAPTER is set but VIZ_QUANTIZE is on: an adapter trained on a "
                "bf16 base misbehaves badly on a 4-bit one (~32 points of success "
                "rate). Re-launch with VIZ_QUANTIZE=0, or unset VIZ_ADAPTER to serve "
                "the untrained base."
            )
        from peft import PeftModel

        print(f"Applying adapter {_ADAPTER} ...", flush=True)
        model = PeftModel.from_pretrained(model, _ADAPTER)
        model.eval()
    # Claude is optional: only wire it up if a key is present, so the server still
    # boots (Gemma-only) without ANTHROPIC_API_KEY.
    anthropic_client = anthropic.Anthropic() if os.environ.get("ANTHROPIC_API_KEY") else None
    _ENGINE.update(
        model=model,
        tokenizer=tokenizer,
        anthropic_client=anthropic_client,
    )
    print(f"Gemma checkpoint: {_CHECKPOINT} ({'4-bit' if _QUANTIZE else 'bf16'})")
    print(f"Agents available: gemma{' + claude' if anthropic_client else ' (claude disabled: no ANTHROPIC_API_KEY)'}")
    print(f"web_search (Brave): {'key set' if has_search_key() else 'NO KEY — set BRAVE_API_KEY'}")
    print("scrape_url (local Chromium): no key needed")
    print("Visualizer ready -> http://127.0.0.1:8000")
    yield


app = FastAPI(title="Menu Visualizer", lifespan=lifespan)
_STATIC = Path(__file__).resolve().parent / "static"


class ExtractRequest(BaseModel):
    query: str
    agent: str = "gemma"  # "gemma" (local model) or "claude" (Anthropic API)
    # Comma-separated dietary restrictions (e.g. "vegetarian, no nuts"); "" means
    # no filtering (the full menu). Slotted into the system prompt per request.
    dietary: str = ""
    # System-prompt variant (see prompts.py): "teacher" carries the source-selection
    # guidance, "student" omits it (for context distillation).
    prompt_variant: str = "teacher"


# Sync def -> FastAPI runs it in a threadpool; _EPISODE_LOCK keeps episodes serial.
@app.post("/api/extract")
def extract(req: ExtractRequest) -> dict:
    query = req.query.strip()
    agent = (req.agent or "gemma").lower()
    variant = (req.prompt_variant or "teacher").lower()
    # Echoed back on every response so the page can label how it was produced.
    meta = {"agent": agent, "prompt_variant": variant}
    if agent == "gemma":
        meta["checkpoint"] = _CHECKPOINT

    def fail(error: str, raw: str = ""):
        return {"ok": False, "error": error, "raw": raw, **meta}

    if not query:
        return fail("Empty query.")
    if agent != "gemma" and agent not in CLAUDE_AGENTS:
        return fail(f"Unknown agent {req.agent!r}.")
    if agent in CLAUDE_AGENTS and _ENGINE.get("anthropic_client") is None:
        return fail("Claude is unavailable: set ANTHROPIC_API_KEY and restart the server.")
    if variant not in ("teacher", "student"):
        return fail(f"Unknown prompt variant {req.prompt_variant!r}.")

    # Resolve (and lazily build/cache) the live tools. A missing search key
    # surfaces here as SystemExit -> a clean error instead of a server crash.
    try:
        tools, registry = _get_tools()
    except SystemExit as e:
        return fail(str(e))

    # Build the prompt with this request's dietary restrictions + variant. (There is
    # no `live=` argument any more: v2 removed the offline stub, so every prompt is
    # the live one. Passing it raised TypeError on EVERY extraction.)
    system_prompt = build_system_prompt(req.dietary, variant=variant)

    # Mark the start of the episode so the buffered tool-call prints below can be
    # attributed to a query/agent. flush=True so it shows immediately even though
    # stdout is block-buffered through uvicorn's pipe.
    print(
        f"\n=== Episode (agent={agent}, variant={variant}, "
        f"dietary={req.dietary!r}): {query!r} ===",
        flush=True,
    )

    with _EPISODE_LOCK:
        if agent == "gemma":
            answer = run_gemma_episode(
                _ENGINE["model"], _ENGINE["tokenizer"], query, tools, registry, system_prompt
            )
        else:
            # claude_agent.run_episode returns (final_text, messages) -- unpack it.
            # Binding the tuple to `answer` fed a tuple to extract_json and made
            # every Claude extraction fail as unparseable.
            answer, _messages = run_claude_episode(
                _ENGINE["anthropic_client"], query, tools, registry, system_prompt,
                model=CLAUDE_AGENTS[agent],
            )

    parsed, err = extract_json(answer)
    if parsed is None:
        return fail(f"Model output was not valid JSON: {err}", raw=answer)
    return {"ok": True, "menu": parsed, "raw": answer, **meta}


@app.get("/api/config")
def config() -> dict:
    """What the server is actually serving, so the page can label + default itself.

    `default_variant` matters: a trained student was distilled under the STUDENT
    prompt and is evaluated under it, so defaulting the UI to "teacher" would be a
    train/serve mismatch and would under-report the model. The raw base has no such
    constraint and keeps the teacher prompt (its guidance is all it has).
    """
    return {
        "checkpoint": _CHECKPOINT,
        "quantized": _QUANTIZE,
        "default_variant": "student" if _ADAPTER else "teacher",
        "claude": _ENGINE.get("anthropic_client") is not None,
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_STATIC / "index.html")
