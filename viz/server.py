"""Local visualizer for the Gemma menu-extraction agent.

A single FastAPI process that:
  - loads the Gemma model + Firecrawl MCP tools ONCE at startup (in-process),
  - serves the static page (static/index.html) at "/",
  - exposes POST /api/extract {"query": "<restaurant>"} -> menu JSON.

The model and the single GPU are a shared singleton, so extraction is serialized
behind a lock: one episode runs at a time no matter how many browser tabs hit it
(concurrent generate() calls on one GPU would race / OOM). FastAPI runs the sync
endpoint in a threadpool, so the lock -- not the event loop -- does the gating.

Run from the repo root:
    uv run uvicorn viz.server:app --host 127.0.0.1 --port 8000

Requires FIRECRAWL_API_KEY in the repo-root .env (the --mcp tool path needs it).
"""

from __future__ import annotations

import os
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

# The src/ modules use flat imports (`from model import ...`, `from agent import
# ...`) and expect both src/ and src/gemma/ on sys.path -- mirror the convention
# run_agent.py sets up so we can reuse the engine untouched.
_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
for _p in (_SRC, _SRC / "gemma"):
    sys.path.insert(0, str(_p))

load_dotenv(_REPO / ".env")  # FIRECRAWL_API_KEY, like run_agent.py

# Imported after sys.path is set. model.py sets PYTORCH_CUDA_ALLOC_CONF before it
# touches torch, so it must be the first of these to import.
from model import load_model       # noqa: E402
from tools import setup_tools      # noqa: E402
from agent import run_episode      # noqa: E402
from schema import extract_json    # noqa: E402

# 4-bit by default on this box (15 GB host RAM; see CLAUDE.md). Set VIZ_QUANTIZE=0
# on a bigger-RAM machine to load full-quality bf16 instead.
_QUANTIZE = os.environ.get("VIZ_QUANTIZE", "1") != "0"

_ENGINE: dict = {}              # model/tokenizer/tools, populated at startup
_GPU_LOCK = threading.Lock()   # serialize generation on the single GPU


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model + Firecrawl tools once; tear the MCP subprocess down on exit."""
    model, tokenizer = load_model(quantize=_QUANTIZE, attn="sdpa")
    tools, registry, system_prompt, mcp_client = setup_tools(use_mcp=True)
    _ENGINE.update(
        model=model,
        tokenizer=tokenizer,
        tools=tools,
        registry=registry,
        system_prompt=system_prompt,
        mcp_client=mcp_client,
    )
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


# Sync def -> FastAPI runs it in a threadpool; _GPU_LOCK keeps episodes serial.
@app.post("/api/extract")
def extract(req: ExtractRequest) -> dict:
    query = req.query.strip()
    if not query:
        return {"ok": False, "error": "Empty query.", "raw": ""}

    with _GPU_LOCK:
        answer = run_episode(
            _ENGINE["model"],
            _ENGINE["tokenizer"],
            query,
            _ENGINE["tools"],
            _ENGINE["registry"],
            _ENGINE["system_prompt"],
        )

    parsed, err = extract_json(answer)
    if parsed is None:
        return {"ok": False, "error": f"Model output was not valid JSON: {err}", "raw": answer}
    return {"ok": True, "menu": parsed, "raw": answer}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_STATIC / "index.html")
