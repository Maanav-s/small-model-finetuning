"""Local trace REVIEW tool for the menu-extraction corpus.

A tiny, read-mostly FastAPI app for reviewing menu-extraction traces one at a
time and marking each **keep** (accept) or **reject**. The rejects become the
reject-list consumed by scripts/build_sft.py (`--reject-list`), which drops those
traces from the SFT dataset.

Unlike viz/server.py, this app loads **no model, no torch, no anthropic, no
tools, no network** -- it only reads/writes small JSON files under `data/`. Keep
it that way: importing it must never pull in the GPU stack.

It reuses viz/server.py's conventions only where they apply here: FileResponse
for the static page and the `_REPO` pathing. (The `src/` flat-import sys.path
dance is intentionally omitted -- this app imports nothing from `src/`.)

Data files (all under DATA_DIR, default <repo>/data):
  - traces/*.json          the traces (contract 1.5); read-only here.
  - review/grounding.json  per-trace %-grounded + the not-found->found sibling map,
                           precomputed by scripts/audit_grounding.py; read-only here.
  - review/decisions.json  the reviewer's keep/reject choices (this app writes it).
  - review/reject_list.txt the exported reject list (this app writes it).

DATA_DIR is a module global so tests can point it at a tmp dir
(`monkeypatch.setattr(viz.review, "DATA_DIR", tmp)`); it also honours the
REVIEW_DATA_DIR env var. Every path is derived from DATA_DIR at call time, so a
monkeypatch after import takes effect.

Run from the repo root:
    uv run uvicorn viz.review:app --host 127.0.0.1 --port 8001
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

_REPO = Path(__file__).resolve().parent.parent
_STATIC = Path(__file__).resolve().parent / "static"

# Overridable root for all data files. Tests monkeypatch this; ops can set
# REVIEW_DATA_DIR. Everything below reads DATA_DIR lazily via the _*_dir helpers.
DATA_DIR = Path(os.environ.get("REVIEW_DATA_DIR", _REPO / "data"))

# How many chars of each tool result to surface in the compact conversation view.
TOOL_RESULT_PREVIEW_CHARS = 800

# Hard cap on the full-tool-result endpoint payload (defends against a pathological
# multi-MB scrape). The reviewer can still see the vast majority of any real page.
TOOL_RESULT_MAX_CHARS = 200_000

# Half-window (chars) shown either side of a matched sibling item name, so the
# reviewer can SEE where in the scraped text the match landed.
SIBLING_CONTEXT_CHARS = 100

# Markers wrapped around the matched span inside a context snippet. The page turns
# these into a <mark> highlight; they are plain unicode so escaping leaves them be.
_MATCH_OPEN = "〈"   # 〈
_MATCH_CLOSE = "〉"  # 〉

# Guard the sibling-evidence panel against an absurdly large menu blob.
SIBLING_MENU_MAX_ITEMS = 300


# --- path helpers (read DATA_DIR at call time so monkeypatching works) --------

def _traces_dir() -> Path:
    return DATA_DIR / "traces"


def _review_dir() -> Path:
    return DATA_DIR / "review"


def _decisions_path() -> Path:
    return _review_dir() / "decisions.json"


def _reject_list_path() -> Path:
    return _review_dir() / "reject_list.txt"


def _grounding_path() -> Path:
    return _review_dir() / "grounding.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --- small JSON IO ------------------------------------------------------------

def _read_json(path: Path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _atomic_write_json(path: Path, obj) -> None:
    """Write JSON atomically (tmp + os.replace) so a crash can't truncate the file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _load_decisions() -> dict:
    d = _read_json(_decisions_path(), {})
    return d if isinstance(d, dict) else {}


def _load_grounding() -> dict:
    """Per-trace grounding + sibling map from scripts/audit_grounding.py, or {}.

    Purely additive: an empty map means "no precomputed grounding", and the app
    degrades to loading traces directly (grounding shown as null, no siblings).
    """
    d = _read_json(_grounding_path(), {})
    return d if isinstance(d, dict) else {}


# --- trace helpers ------------------------------------------------------------

def _n_items(final_json: dict) -> int:
    menu = final_json.get("menu") if isinstance(final_json, dict) else None
    if not isinstance(menu, list):
        return 0
    total = 0
    for section in menu:
        items = section.get("items") if isinstance(section, dict) else None
        if isinstance(items, list):
            total += len(items)
    return total


def _load_trace(filename: str) -> dict | None:
    """Load one trace by filename. Returns None if missing or path-unsafe."""
    # Guard against traversal: only a bare filename in traces/ is allowed.
    if "/" in filename or "\\" in filename or filename in ("", ".", ".."):
        return None
    path = _traces_dir() / filename
    if not path.is_file():
        return None
    return _read_json(path, None)


def _trace_grounding(trace: dict) -> float | None:
    """Live %-grounded for a trace: fraction of its own menu item names that appear
    in its own scraped/searched text. None when the trace has no menu items.

    Uses the SAME matcher as the sibling panel (_match_menu_items) so the primary
    trace's % and its (would-be) per-item breakdown always agree, and agrees with
    scripts/audit_grounding.py's precomputed value for the list.
    """
    fj = trace.get("final_json") or {}
    matches = _match_menu_items(fj.get("menu"), _sibling_tool_segments(trace))
    if not matches:
        return None
    matched = sum(1 for m in matches if m["matched"])
    return round(matched / len(matches), 3)


def _summary_from_grounding(filename: str, g: dict, decisions: dict) -> dict:
    """A nav/list summary built purely from a grounding.json entry (no trace load)."""
    dec = decisions.get(filename)
    return {
        "filename": filename,
        "restaurant_name": g.get("restaurant_name") or "",
        "episode_input": g.get("episode_input") or "",
        "dietary_restrictions": g.get("restrictions"),
        "found": bool(g.get("found")),
        "schema_valid": bool(g.get("schema_valid")),
        "n_items": int(g.get("n_items") or 0),
        "grounding": g.get("grounding"),
        "decision": (dec or {}).get("decision"),
    }


def _summary_from_trace(filename: str, trace: dict, decisions: dict) -> dict:
    """Fallback nav/list summary when grounding.json is absent (loads the trace)."""
    fj = trace.get("final_json") or {}
    dec = decisions.get(filename)
    return {
        "filename": filename,
        "restaurant_name": trace.get("restaurant_name") or fj.get("restaurant_name") or "",
        "episode_input": trace.get("episode_input") or "",
        "dietary_restrictions": trace.get("dietary_restrictions"),
        "found": bool(fj.get("found")),
        "schema_valid": bool(trace.get("schema_valid")),
        "n_items": _n_items(fj),
        "grounding": None,
        "decision": (dec or {}).get("decision"),
    }


def _trim_menu(menu) -> list:
    """Cap a sibling menu to SIBLING_MENU_MAX_ITEMS items so the panel stays sane."""
    if not isinstance(menu, list):
        return []
    out: list = []
    remaining = SIBLING_MENU_MAX_ITEMS
    for section in menu:
        if remaining <= 0 or not isinstance(section, dict):
            break
        items = section.get("items") if isinstance(section.get("items"), list) else []
        items = items[:remaining]
        remaining -= len(items)
        out.append({"section": section.get("section") or "", "items": items})
    return out


def _siblings_for(filename: str, grounding_map: dict, decisions: dict) -> list:
    """Sibling entries for a trace: the other slices of the same restaurant (free +
    dietary-conditioned, from grounding.json's sibling map), each with its
    found/%-grounded state and its rendered menu (loaded from its trace file) so
    the reviewer can compare slices and cross-check identity.

    Each entry carries the sibling's own stored `decision` (keep|reject|None) so
    the panel can render its state on load -- a sibling is itself a real trace
    file, so it uses the same decisions.json keyed by its filename.
    """
    sib_files = (grounding_map.get(filename) or {}).get("siblings") or []
    out: list = []
    for sib_file in sib_files:
        if not isinstance(sib_file, str):
            continue
        g = grounding_map.get(sib_file) or {}
        sib_trace = _load_trace(sib_file)
        sib_fj = (sib_trace or {}).get("final_json") or {}
        trimmed = _trim_menu(sib_fj.get("menu"))
        segments = _sibling_tool_segments(sib_trace) if sib_trace else []
        out.append({
            "sibling_file": sib_file,
            "restaurant_name": g.get("restaurant_name") or (sib_trace or {}).get("restaurant_name"),
            "restrictions": g.get("restrictions"),
            "found": bool(g.get("found") if g else sib_fj.get("found")),
            "grounding": g.get("grounding"),
            "n_items": g.get("n_items"),
            "source_url": g.get("source_url") or sib_fj.get("source_url"),
            "unmatched_items": g.get("unmatched_items") or [],
            "menu": trimmed,
            # The sibling's own keep/reject/None state (shown as a badge + controls).
            "decision": (decisions.get(sib_file) or {}).get("decision"),
            # Per-item: did this item's name appear in the sibling's scraped text,
            # and if so a ±context window with the match marked (the "where").
            "items": _match_menu_items(trimmed, segments),
        })
    # Found slices first, then most-suspicious (lowest grounding) among them; the
    # abstained (not-found) siblings sort to the end.
    out.sort(key=lambda s: (not s.get("found"),
                            s.get("grounding") if isinstance(s.get("grounding"), (int, float)) else 1.0))
    return out


def _aggregate(summaries: list[dict]) -> dict:
    kept = sum(1 for s in summaries if s["decision"] == "keep")
    rejected = sum(1 for s in summaries if s["decision"] == "reject")
    total = len(summaries)
    return {
        "total": total,
        "kept": kept,
        "rejected": rejected,
        "undecided": total - kept - rejected,
    }


def _tool_result_text(block: dict) -> str:
    """Flatten an Anthropic tool_result block's content to a string."""
    content = block.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for piece in content:
            if isinstance(piece, dict):
                parts.append(piece.get("text", ""))
            else:
                parts.append(str(piece))
        return "\n".join(parts)
    return "" if content is None else str(content)


def _sibling_tool_segments(trace: dict) -> list[tuple[str, str | None]]:
    """(raw_text, source_url|None) for each tool_result in a trace, in order.

    source_url is the URL of the `scrape_url` call the result answers (resolved via
    tool_use_id -> that tool_use's `input.url`), or None for a search result. Used
    to tell the reviewer WHICH scrape a matched item came from.
    """
    messages = trace.get("messages") or []
    id_to_url: dict[str, str | None] = {}
    for m in messages:
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                bid = block.get("id")
                inp = block.get("input")
                url = inp.get("url") if isinstance(inp, dict) else None
                if bid:
                    id_to_url[bid] = url if isinstance(url, str) else None
    segments: list[tuple[str, str | None]] = []
    for m in messages:
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                segments.append(
                    (_tool_result_text(block), id_to_url.get(block.get("tool_use_id")))
                )
    return segments


def _match_menu_items(menu: list, segments: list) -> list[dict]:
    """Per menu item: does its name appear in the concatenated tool text, and where.

    Mirrors scripts/audit_grounding.py's match EXACTLY (lowercase, collapse runs of
    whitespace to one space, strip, case-insensitive substring). For a match,
    `context` is a ±SIBLING_CONTEXT_CHARS window around the FIRST occurrence with
    the matched span wrapped in _MATCH_OPEN/_MATCH_CLOSE and whitespace collapsed to
    one line; `source_hint` is the scrape URL whose result contains the match.

    Returns [{name, matched: bool, context: str|None, source_hint: str|None}].
    """
    raw = "\n".join(seg for seg, _ in segments)
    low = raw.lower()  # positions align with `raw`: "\n" join char lowercases to itself
    # Segment offset spans (incl. the 1-char "\n" join), for source_hint lookup.
    spans: list[tuple[int, int, str | None]] = []
    pos = 0
    for seg, url in segments:
        spans.append((pos, pos + len(seg), url))
        pos += len(seg) + 1

    out: list[dict] = []
    for section in menu or []:
        if not isinstance(section, dict):
            continue
        for it in section.get("items") or []:
            name = (it.get("name") or "").strip() if isinstance(it, dict) else ""
            if not name:
                continue
            core = re.sub(r"\s+", " ", name.lower()).strip()
            idx = low.find(core) if core else -1
            if idx < 0:
                out.append({"name": name, "matched": False,
                            "context": None, "source_hint": None})
                continue
            start = max(0, idx - SIBLING_CONTEXT_CHARS)
            end = min(len(raw), idx + len(core) + SIBLING_CONTEXT_CHARS)
            snippet = (raw[start:idx] + _MATCH_OPEN + raw[idx:idx + len(core)]
                       + _MATCH_CLOSE + raw[idx + len(core):end])
            snippet = re.sub(r"\s+", " ", snippet).strip()
            if start > 0:
                snippet = "…" + snippet
            if end < len(raw):
                snippet = snippet + "…"
            hint = None
            for s0, s1, url in spans:
                if s0 <= idx < s1:
                    hint = url
                    break
            out.append({"name": name, "matched": True,
                        "context": snippet, "source_hint": hint})
    return out


def _walk_turns(messages: list) -> list[dict]:
    """Turn-by-turn view of the conversation carrying FULL tool-result text.

    Shared spine for both the compact preview (which truncates) and the
    full-tool-result endpoint (which addresses a block by its turn/idx here), so
    the two agree on turn ordering. Wholly-empty turns are dropped from both.
    """
    turns: list[dict] = []
    for msg in messages or []:
        role = msg.get("role")
        content = msg.get("content")
        turn: dict = {"role": role, "text": "", "tool_calls": [], "tool_results": []}
        if isinstance(content, str):
            turn["text"] = content
        elif isinstance(content, list):
            texts = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    texts.append(block.get("text", ""))
                elif btype == "tool_use":
                    turn["tool_calls"].append(
                        {"name": block.get("name"), "input": block.get("input")}
                    )
                elif btype == "tool_result":
                    raw = _tool_result_text(block)
                    turn["tool_results"].append({"text": raw, "full_len": len(raw)})
            turn["text"] = "\n".join(t for t in texts if t)
        if turn["text"] or turn["tool_calls"] or turn["tool_results"]:
            turns.append(turn)
    return turns


def _compact_conversation(messages: list) -> list[dict]:
    """A small, reviewer-friendly view of the Anthropic conversation.

    Each entry is a turn with role and, depending on role, any assistant text/
    tool calls or truncated tool-result previews -- enough to see what the agent
    did and saw without dumping 75k-char scraped pages. Each tool-result preview
    carries its `turn`/`idx` address so the page can fetch the full text on demand
    (GET /api/review/toolresult/{filename}?turn=&idx=).
    """
    out: list[dict] = []
    for ti, turn in enumerate(_walk_turns(messages)):
        results = []
        for bi, tr in enumerate(turn["tool_results"]):
            raw = tr["text"]
            results.append({
                "turn": ti,
                "idx": bi,
                "preview": raw[:TOOL_RESULT_PREVIEW_CHARS],
                "truncated": len(raw) > TOOL_RESULT_PREVIEW_CHARS,
                "full_len": len(raw),
            })
        out.append({
            "role": turn["role"],
            "text": turn["text"],
            "tool_calls": turn["tool_calls"],
            "tool_results": results,
        })
    return out


# --- app ----------------------------------------------------------------------

app = FastAPI(title="Trace Review")


class DecisionRequest(BaseModel):
    filename: str
    # "keep" | "reject" to set; None or "undo" to clear.
    decision: str | None = None


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_STATIC / "review.html")


@app.get("/api/review/traces")
def list_traces(scope: str = "notfound") -> dict:
    """Ordered summaries for the nav + an aggregate.

    scope=notfound (default): only found=false traces -- the review task.
    scope=all: every trace under traces/.
    Sorted not-found first, then by filename. Each summary carries the trace's
    %-grounded stat (from grounding.json) when available.
    """
    scope = (scope or "notfound").lower()
    decisions = _load_decisions()
    grounding_map = _load_grounding()

    summaries: list[dict] = []
    if grounding_map:
        # Fast path: build entirely from the precomputed index -- no trace loads.
        for filename in sorted(grounding_map.keys()):
            g = grounding_map[filename]
            if not isinstance(g, dict):
                continue
            if scope != "all" and bool(g.get("found")):
                continue
            summaries.append(_summary_from_grounding(filename, g, decisions))
    else:
        # Fallback: no grounding.json yet -- glob and load traces directly.
        for filename in sorted(p.name for p in _traces_dir().glob("*.json")):
            trace = _load_trace(filename)
            if trace is None:
                continue
            found = bool((trace.get("final_json") or {}).get("found"))
            if scope != "all" and found:
                continue
            summaries.append(_summary_from_trace(filename, trace, decisions))

    # Not-found first, then filename (stable within each group).
    summaries.sort(key=lambda s: (s["found"], s["filename"]))

    return {"scope": scope, "traces": summaries, "aggregate": _aggregate(summaries)}


@app.get("/api/review/trace/{filename}")
def get_trace(filename: str) -> dict:
    trace = _load_trace(filename)
    if trace is None:
        return {"ok": False, "error": f"Trace not found: {filename!r}"}
    decisions = _load_decisions()
    grounding_map = _load_grounding()
    fj = trace.get("final_json") or {}
    dec = decisions.get(filename)
    siblings = _siblings_for(filename, grounding_map, decisions)

    resp = {
        "ok": True,
        "filename": filename,
        "restaurant_name": trace.get("restaurant_name") or fj.get("restaurant_name") or "",
        "episode_input": trace.get("episode_input") or "",
        "dietary_restrictions": trace.get("dietary_restrictions"),
        "found": bool(fj.get("found")),
        "schema_valid": bool(trace.get("schema_valid")),
        "n_items": _n_items(fj),
        # %-grounded for THIS trace: how much of its own menu is in what it scraped.
        "grounding": _trace_grounding(trace),
        "final_json": fj,
        "menu": fj.get("menu") or [],
        "notes": fj.get("notes"),
        "source_url": fj.get("source_url"),
        "queries": trace.get("queries") or [],
        "urls": trace.get("urls") or [],
        "decision": (dec or {}).get("decision"),
        "conversation": _compact_conversation(trace.get("messages") or []),
    }
    if siblings:
        resp["siblings"] = siblings
    return resp


@app.get("/api/review/toolresult/{filename}")
def get_tool_result(filename: str, turn: int = 0, idx: int = 0) -> dict:
    """Full (untruncated, capped at TOOL_RESULT_MAX_CHARS) text of one tool_result.

    Addressed by (turn, idx) into the same _walk_turns ordering the compact
    conversation exposes, so the page can expand a preview on demand while default
    payloads stay small. Path traversal is guarded by _load_trace (bare filename).
    """
    trace = _load_trace(filename)
    if trace is None:
        return {"ok": False, "error": f"Trace not found: {filename!r}"}
    turns = _walk_turns(trace.get("messages") or [])
    if turn < 0 or turn >= len(turns):
        return {"ok": False, "error": f"turn {turn} out of range (0..{len(turns) - 1})"}
    results = turns[turn]["tool_results"]
    if idx < 0 or idx >= len(results):
        return {"ok": False, "error": f"idx {idx} out of range (0..{len(results) - 1})"}
    raw = results[idx]["text"]
    return {
        "ok": True,
        "filename": filename,
        "turn": turn,
        "idx": idx,
        "text": raw[:TOOL_RESULT_MAX_CHARS],
        "full_len": len(raw),
        "truncated": len(raw) > TOOL_RESULT_MAX_CHARS,
    }


def _scope_aggregate(scope: str) -> dict:
    """Aggregate over the current scope, for the response of a decision write."""
    return list_traces(scope=scope)["aggregate"]


@app.post("/api/review/decision")
def post_decision(req: DecisionRequest, scope: str = "notfound") -> dict:
    filename = req.filename
    if _load_trace(filename) is None:
        return {"ok": False, "error": f"Trace not found: {filename!r}"}

    decisions = _load_decisions()
    choice = req.decision
    if choice in (None, "undo", "", "null"):
        decisions.pop(filename, None)
        new_decision = None
    elif choice in ("keep", "reject"):
        decisions[filename] = {"decision": choice, "at": _now_iso()}
        new_decision = choice
    else:
        return {"ok": False, "error": f"Invalid decision: {choice!r} (want keep|reject|undo)"}

    _atomic_write_json(_decisions_path(), decisions)
    return {"ok": True, "filename": filename, "decision": new_decision,
            "aggregate": _scope_aggregate(scope)}


def _write_reject_list() -> dict:
    # Iterate the WHOLE decisions map (not a scope's traces): a rejected sibling is
    # a found=true trace outside the not-found review scope, but its filename must
    # still land in reject_list.txt so build_sft.py drops it.
    decisions = _load_decisions()
    rejects = sorted(fn for fn, d in decisions.items()
                     if isinstance(d, dict) and d.get("decision") == "reject")
    path = _reject_list_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    header = f"# generated {_now_iso()} — {len(rejects)} rejects\n"
    body = "".join(fn + "\n" for fn in rejects)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(header)
        f.write(body)
    os.replace(tmp, path)
    return {"ok": True, "path": str(path), "count": len(rejects), "rejects": rejects}


@app.post("/api/review/export")
def post_export() -> dict:
    return _write_reject_list()


@app.get("/api/review/export")
def get_export() -> dict:
    return _write_reject_list()
