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
  model and an Anthropic client **once** (in-process, via `src/`'s `load_model`
  and `anthropic.Anthropic`) and keeps them resident. The web tools (Brave search
  + Jina scrape) are built **lazily once and cached** (via `setup_tools`), so the
  server boots with no web key and only needs the keys the first time you run an
  extraction. It exposes:
  - `GET /` → serves `static/index.html`
  - `POST /api/extract`
    `{"query": "<restaurant>", "agent": "gemma"|"claude", "dietary": "<optional>",
    "prompt_variant": "teacher"|"student"}`
    → runs one extraction episode and returns
    `{"ok": true, "menu": {...}, "raw": "...", "agent": "...", "prompt_variant": "..."}`,
    or `{"ok": false, "error": "...", ...}` on a missing key / non-JSON output.
    `agent` defaults to `gemma`. `dietary` is an optional comma-separated list of
    dietary restrictions (e.g. `"vegetarian, no nuts"`); it's slotted into the
    system prompt per request so the model filters the menu to complying items,
    and an empty string means no filtering (the full menu). `prompt_variant`
    defaults to `teacher` (see "Choosing the prompt variant"). Validation reuses
    `schema.extract_json`, so the page and the eval/reward share one contract.

### Dietary restrictions & failure handling

- **Scope** — the prompt always returns just the **food dishes**
  (appetizers, mains, shared plates, sides, desserts) and drops menu bulk that
  isn't a dish — drinks/beverages, add-ons, modifiers, upsells, merch — so a long
  menu's drink list and extras don't crowd out what the diner can actually eat.
- **Dietary filter** — a second input on the page takes comma-separated
  restrictions. They're built into the system prompt (`build_system_prompt`), so
  the model returns only complying dishes; blank = every food dish (still no
  drinks/bulk). If the filter leaves nothing, the page shows a *"No matching menu
  items"* notice (distinct from a not-found menu).
- **Menu not found** — when the model can't find a menu at all it returns the
  `found: false` shape (`schema.NOT_FOUND_SNIPPET`) with a short `notes`; the page
  renders a dedicated "No menu found" card instead of an empty menu.
- **Missing prices** — an item with no discoverable price has `price: null`
  (`schema.PRICE_UNKNOWN`; the model is told never to guess one), which the page
  renders as a muted *"no price"* marker rather than a blank that reads as free.

### Choosing the agent

The page has a dropdown to pick **Gemma (local)** or **Claude (API)**; both run
the *same* tools, system prompt, and JSON contract — only the loop differs
(`gemma/agent.py` vs `claude/claude_agent.py`), so the two are directly
comparable. The rendered menu shows which agent produced it.

Claude is optional: it's wired up only if `ANTHROPIC_API_KEY` is present at
startup. Without the key the server still boots Gemma-only, and a Claude request
returns an error explaining the key is missing (rather than failing at startup).

### Choosing the prompt variant

A third dropdown picks the system-prompt variant: **Teacher** (default) or
**Student**. They differ only by a block of source-selection guidance (prefer the
restaurant's own site, avoid delivery apps) that the teacher includes and the
student omits — the setup for context distillation (see the repo `CLAUDE.md`).
Teacher is what we test on and generate SFT data with; switch to Student to
preview the prompt the distilled model is trained to run under. The rendered menu
labels which variant produced it.

The web tools are fixed: **Brave** backs `web_search` and **Jina** backs
`scrape_url` (see [src/backends.py](../src/backends.py)).

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
singleton, and concurrent `generate()` calls would race or OOM. The sync
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
- Tool calls use Brave (search) + Jina (scrape). Set `BRAVE_API_KEY` and
  `JINA_API_KEY` in the repo-root `.env` (see [.env.example](../.env.example));
  the startup log prints whether each key is set.
- To enable the **Claude** agent, add `ANTHROPIC_API_KEY` to the repo-root `.env`
  (same key `run_claude.py` uses). Gemma needs no extra key.

### Notes / knobs

- **Quantization:** 4-bit by default (matches this 15 GB-host-RAM box — see the
  repo `CLAUDE.md`). Set `VIZ_QUANTIZE=0` to load full-quality bf16 on a
  bigger-RAM machine.
- **Latency:** a request runs a full episode — Brave search / Jina scrape plus up
  to several generation turns — so it can take a couple of minutes, especially in
  4-bit. The page shows a loading state; expect minutes, not milliseconds. If you
  script against the API with `curl`, set a generous `--max-time`.
- **Remote box:** bind `--host 0.0.0.0` to reach it from your laptop to the GPU
  instance (otherwise it's localhost-only).
