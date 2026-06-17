# viz — local menu visualizer

A tiny, self-contained web app that puts a face on the Phase 1 agent loop: type a
restaurant name, the Gemma model runs the same `run_episode` the CLI uses, and the
returned menu JSON is rendered as a styled menu page in the browser.

This folder is **demo/visualization only**. It imports the engine from `src/` but
adds nothing the training/eval pipeline depends on — deleting `viz/` leaves the
rest of the project untouched.

## Layout

```
viz/
  server.py          # FastAPI backend: loads the model once, serves the API + page
  static/index.html  # frontend: query box -> fetch -> rendered menu (vanilla JS)
  README.md          # this file
```

## How it works

One Python process does everything — there is no separate frontend server.

- **Backend ([server.py](server.py))** — FastAPI. On startup it loads the Gemma
  model, an Anthropic client, and the Firecrawl MCP tools **once** (in-process,
  via `src/`'s `load_model`, `anthropic.Anthropic`, and `setup_tools`) and keeps
  them resident. It exposes:
  - `GET /` → serves `static/index.html`
  - `POST /api/extract` `{"query": "<restaurant>", "agent": "gemma"|"claude"}` →
    runs one extraction episode with the chosen agent and returns
    `{"ok": true, "menu": {...}, "raw": "...", "agent": "..."}`, or
    `{"ok": false, "error": "...", "raw": "...", "agent": "..."}` when the
    model's output isn't valid JSON. `agent` defaults to `gemma`. Validation
    reuses `schema.extract_json`, so the page and the eval/reward share one
    contract.

### Choosing the agent

The page has a dropdown to pick **Gemma (local)** or **Claude (API)**; both run
the *same* tools, system prompt, and JSON contract — only the loop differs
(`gemma/agent.py` vs `claude/claude_agent.py`), so the two are directly
comparable. The rendered menu shows which agent produced it.

Claude is optional: it's wired up only if `ANTHROPIC_API_KEY` is present at
startup. Without the key the server still boots Gemma-only, and a Claude request
returns an error explaining the key is missing (rather than failing at startup).
- **Frontend ([static/index.html](static/index.html))** — plain HTML + vanilla
  JS, no framework and no build step. It `fetch()`es `/api/extract`, then renders
  `menu[].section` / `items[].{name, description, price}` as a menu card (dotted
  price leaders, italic descriptions, a source link). The model returns *data*;
  the JS is what turns that data into a page. Unparseable output falls back to
  showing the raw text.

### Why one process, and why serialized

`server.py` wraps every episode in a single `threading.Lock`, so they run **one
at a time** no matter how many browser tabs (or which agent) hit the endpoint.
For Gemma this is essential — the model and the single dev GPU are a shared
singleton, and concurrent `generate()` calls would race or OOM. The lock also
guards the shared Firecrawl MCP subprocess that *both* agents call. The sync
endpoint runs in FastAPI's threadpool, so the lock (not the event loop) does the
gating. This is fine for a local demo; it is not a multi-tenant server. (A
consequence: a slow Gemma run will block a concurrent Claude request until it
finishes.)

## Running it

From the repo root:

```bash
uv run uvicorn viz.server:app --host 127.0.0.1 --port 8000
```

Then open <http://127.0.0.1:8000>.

- Startup loads the model first (tens of seconds to ~1.5 min on this box); the
  page won't serve until you see `Visualizer ready -> ...` in the log.
- Tool calls use the **live Firecrawl MCP server**, so `FIRECRAWL_API_KEY` must be
  in the repo-root `.env` (same requirement as `run_agent.py --mcp`), and
  Node/`npx` must be on `PATH`. The server spins up `npx -y firecrawl-mcp` at
  startup and tears it down on shutdown.
- To enable the **Claude** agent, add `ANTHROPIC_API_KEY` to the repo-root `.env`
  (same key `run_claude.py` uses). Gemma needs no extra key.

### Notes / knobs

- **Quantization:** 4-bit by default (matches this 15 GB-host-RAM box — see the
  repo `CLAUDE.md`). Set `VIZ_QUANTIZE=0` to load full-quality bf16 on a
  bigger-RAM machine.
- **Latency:** a request runs a full episode — Firecrawl search/scrape plus up to
  several generation turns — so it can take a couple of minutes, especially in
  4-bit. The page shows a loading state; expect minutes, not milliseconds. If you
  script against the API with `curl`, set a generous `--max-time`.
- **Remote box:** bind `--host 0.0.0.0` to reach it from your laptop to the GPU
  instance (otherwise it's localhost-only).
