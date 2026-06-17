"""Local visualizer for the Gemma menu-extraction agent.

A single FastAPI process that:
  - loads the Gemma model + an Anthropic client + Firecrawl MCP tools ONCE at
    startup (in-process),
  - serves the static page (static/index.html) at "/",
  - exposes POST /api/extract {"query": "<restaurant>", "agent": "gemma"|"claude"}
    -> menu JSON.

Either agent runs the SAME tools / system prompt / JSON contract (only the loop
differs): the local Gemma model (gemma/agent.py) or Claude via the Anthropic API
(claude/claude_agent.py). The default is gemma.

Episodes are serialized behind one lock: one runs at a time no matter how many
browser tabs hit it. For Gemma this is essential (concurrent generate() calls on
the single GPU would race / OOM); it also guards the shared Firecrawl MCP
subprocess that both agents call. FastAPI runs the sync endpoint in a threadpool,
so the lock -- not the event loop -- does the gating.

Run from the repo root:
    uv run uvicorn viz.server:app --host 127.0.0.1 --port 8000

Requires FIRECRAWL_API_KEY in the repo-root .env (the MCP tool path needs it).
The Claude agent additionally needs ANTHROPIC_API_KEY; without it, only Gemma is
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

load_dotenv(_REPO / ".env")  # FIRECRAWL_API_KEY (+ ANTHROPIC_API_KEY for Claude)

# Imported after sys.path is set. model.py sets PYTORCH_CUDA_ALLOC_CONF before it
# touches torch, so it must be the first of these to import. The two run_episode
# loops share a name, so alias them.
from model import load_model                          # noqa: E402
from tools import setup_tools                          # noqa: E402
from agent import run_episode as run_gemma_episode     # noqa: E402
from claude_agent import run_episode as run_claude_episode  # noqa: E402
from schema import extract_json                        # noqa: E402

# 4-bit by default on this box (15 GB host RAM; see CLAUDE.md). Set VIZ_QUANTIZE=0
# on a bigger-RAM machine to load full-quality bf16 instead.
_QUANTIZE = os.environ.get("VIZ_QUANTIZE", "1") != "0"

_ENGINE: dict = {}                 # model/tokenizer/tools/client, populated at startup
_EPISODE_LOCK = threading.Lock()   # serialize episodes (GPU + shared MCP subprocess)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model, Anthropic client, and Firecrawl tools once; close MCP on exit."""
    model, tokenizer = load_model(quantize=_QUANTIZE, attn="sdpa")
    tools, registry, system_prompt, mcp_client = setup_tools(use_mcp=True)
    # Claude is optional: only wire it up if a key is present, so the server still
    # boots (Gemma-only) without ANTHROPIC_API_KEY.
    anthropic_client = anthropic.Anthropic() if os.environ.get("ANTHROPIC_API_KEY") else None
    _ENGINE.update(
        model=model,
        tokenizer=tokenizer,
        tools=tools,
        registry=registry,
        system_prompt=system_prompt,
        mcp_client=mcp_client,
        anthropic_client=anthropic_client,
    )
    print(f"Agents available: gemma{' + claude' if anthropic_client else ' (claude disabled: no ANTHROPIC_API_KEY)'}")
    print("Visualizer ready -> http://127.0.0.1:8000")
    try:
        yield
    finally:
        if mcp_client is not None:
            mcp_client.close()


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
    if not query:
        return {"ok": False, "error": "Empty query.", "raw": "", "agent": agent}
    if agent not in ("gemma", "claude"):
        return {"ok": False, "error": f"Unknown agent {req.agent!r}.", "raw": "", "agent": agent}
    if agent == "claude" and _ENGINE.get("anthropic_client") is None:
        return {
            "ok": False,
            "error": "Claude is unavailable: set ANTHROPIC_API_KEY and restart the server.",
            "raw": "",
            "agent": agent,
        }

    # Mark the start of the episode so the buffered tool-call prints below can be
    # attributed to a query/agent. flush=True so it shows immediately even though
    # stdout is block-buffered through uvicorn's pipe.
    print(f"\n=== Episode (agent={agent}): {query!r} ===", flush=True)

    with _EPISODE_LOCK:
        if agent == "gemma":
            answer = run_gemma_episode(
                _ENGINE["model"],
                _ENGINE["tokenizer"],
                query,
                _ENGINE["tools"],
                _ENGINE["registry"],
                _ENGINE["system_prompt"],
            )
        else:
            answer = run_claude_episode(
                _ENGINE["anthropic_client"],
                query,
                _ENGINE["tools"],
                _ENGINE["registry"],
                _ENGINE["system_prompt"],
            )

    parsed, err = extract_json(answer)
    if parsed is None:
        return {"ok": False, "error": f"Model output was not valid JSON: {err}", "raw": answer, "agent": agent}
    return {"ok": True, "menu": parsed, "raw": answer, "agent": agent}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_STATIC / "index.html")
