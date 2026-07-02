"""Local visualizer for the Gemma menu-extraction agent.

A single FastAPI process that:
  - loads the Gemma model + an Anthropic client ONCE at startup (in-process),
  - serves the static page (static/index.html) at "/", and
  - exposes POST /api/extract {"query", "agent": "gemma"|"claude"} -> menu JSON.

Either agent runs the SAME tools / system prompt / JSON contract (only the loop
differs): the local Gemma model (gemma/agent.py) or Claude via the Anthropic API
(claude/claude_agent.py). The default is gemma.

The web tools are Brave (search) + Jina (scrape) -- see src/backends.py -- built
once at startup.

Episodes are serialized behind one lock: one runs at a time no matter how many
browser tabs hit it. For Gemma this is essential (concurrent generate() calls on
the single GPU would race / OOM). FastAPI runs the sync endpoint in a threadpool,
so the lock -- not the event loop -- does the gating.

Run from the repo root:
    uv run uvicorn viz.server:app --host 127.0.0.1 --port 8000

Set BRAVE_API_KEY and JINA_API_KEY in the repo-root .env for the live tools. The
Claude agent additionally needs ANTHROPIC_API_KEY; without it, only Gemma is
offered (a Claude request returns an error rather than failing at startup).
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

load_dotenv(_REPO / ".env")  # BRAVE_API_KEY / JINA_API_KEY (+ ANTHROPIC_API_KEY for Claude)

# Imported after sys.path is set. model.py sets PYTORCH_CUDA_ALLOC_CONF before it
# touches torch, so it must be the first of these to import. The two run_episode
# loops share a name, so alias them.
from model import load_model                          # noqa: E402
from backends import has_scrape_key, has_search_key   # noqa: E402
from tools import setup_tools                          # noqa: E402
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

_ENGINE: dict = {}                 # model/tokenizer/client, populated at startup
_EPISODE_LOCK = threading.Lock()   # serialize episodes (single GPU: concurrent generate() would race/OOM)

# The live tools (Brave + Jina) are built lazily once and cached here, so the
# server boots with no web key and the key is only needed the first time an
# extraction runs. _TOOLS_LOCK guards the cache against concurrent first builds.
_TOOLS_CACHE: list = []
_TOOLS_LOCK = threading.Lock()


def _get_tools() -> tuple:
    """Return (tools, registry, system_prompt), building+caching once.

    Raises SystemExit (from setup_tools) if a web key is missing; the caller turns
    that into a clean API error instead of crashing the server.
    """
    with _TOOLS_LOCK:
        if not _TOOLS_CACHE:
            _TOOLS_CACHE.append(setup_tools())
        return _TOOLS_CACHE[0]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model and Anthropic client once at startup (tools are built lazily)."""
    model, tokenizer = load_model(quantize=_QUANTIZE, attn="sdpa")
    # Claude is optional: only wire it up if a key is present, so the server still
    # boots (Gemma-only) without ANTHROPIC_API_KEY.
    anthropic_client = anthropic.Anthropic() if os.environ.get("ANTHROPIC_API_KEY") else None
    _ENGINE.update(
        model=model,
        tokenizer=tokenizer,
        anthropic_client=anthropic_client,
    )
    print(f"Agents available: gemma{' + claude' if anthropic_client else ' (claude disabled: no ANTHROPIC_API_KEY)'}")
    print(f"web_search (Brave): {'key set' if has_search_key() else 'NO KEY — set BRAVE_API_KEY'}")
    print(f"scrape_url (Jina): {'key set' if has_scrape_key() else 'NO KEY — set JINA_API_KEY'}")
    print("Visualizer ready -> http://127.0.0.1:8000")
    yield


app = FastAPI(title="Menu Visualizer", lifespan=lifespan)
_STATIC = Path(__file__).resolve().parent / "static"


class ExtractRequest(BaseModel):
    query: str
    agent: str = "gemma"  # "gemma" (local model) or "claude" (Anthropic API)


# Sync def -> FastAPI runs it in a threadpool; _EPISODE_LOCK keeps episodes serial.
@app.post("/api/extract")
def extract(req: ExtractRequest) -> dict:
    query = req.query.strip()
    agent = (req.agent or "gemma").lower()
    # Echoed back on every response so the page can label which agent produced it.
    meta = {"agent": agent}

    def fail(error: str, raw: str = ""):
        return {"ok": False, "error": error, "raw": raw, **meta}

    if not query:
        return fail("Empty query.")
    if agent != "gemma" and agent not in CLAUDE_AGENTS:
        return fail(f"Unknown agent {req.agent!r}.")
    if agent in CLAUDE_AGENTS and _ENGINE.get("anthropic_client") is None:
        return fail("Claude is unavailable: set ANTHROPIC_API_KEY and restart the server.")

    # Resolve (and lazily build/cache) the live tools. A missing key surfaces
    # here as SystemExit -> a clean error instead of a server crash.
    try:
        tools, registry, system_prompt = _get_tools()
    except SystemExit as e:
        return fail(str(e))

    # Mark the start of the episode so the buffered tool-call prints below can be
    # attributed to a query/agent. flush=True so it shows immediately even though
    # stdout is block-buffered through uvicorn's pipe.
    print(f"\n=== Episode (agent={agent}): {query!r} ===", flush=True)

    with _EPISODE_LOCK:
        if agent == "gemma":
            answer = run_gemma_episode(
                _ENGINE["model"], _ENGINE["tokenizer"], query, tools, registry, system_prompt
            )
        else:
            answer = run_claude_episode(
                _ENGINE["anthropic_client"], query, tools, registry, system_prompt,
                model=CLAUDE_AGENTS[agent],
            )

    parsed, err = extract_json(answer)
    if parsed is None:
        return fail(f"Model output was not valid JSON: {err}", raw=answer)
    return {"ok": True, "menu": parsed, "raw": answer, **meta}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_STATIC / "index.html")
