"""Local trace REVIEW tool for the menu-extraction corpus (v2: corpus.sqlite).

A tiny, read-mostly FastAPI app for reviewing menu-extraction traces one at a
time and marking each **keep** (accept) or **reject**. In v2 the rejection is a
DB field (`traces.rejected`), read straight back by scripts/datasets/build_sft.py
(`iter_traces(split='sft', include_rejected=False)`) -- there is no reject-list
file to export anymore.

Unlike viz/server.py, this app loads **no model, no torch, no anthropic, no
tools, no network** -- it only reads/writes the small `corpus.sqlite` via
`src/corpus.py`. Keep it that way: importing it must never pull in the GPU stack
(corpus + grounding are stdlib-only).

The data source moved from the v1 loose files (`data/traces/*.json` +
`data/review/grounding.json` + `data/review/decisions.json` +
`data/review/reject_list.txt`) to the single v2 `corpus.sqlite`:
  - a trace is addressed by its **trace_id** ('<rid>' / '<rid>__<diet-slug>'),
    NOT a filename -- the '.json' is gone;
  - grounding + unmatched_items are trace FIELDS (computed at capture time);
  - the sibling map is a query (`Corpus.siblings`);
  - keep/reject is `Corpus.set_review_decision` (stamps reviewed_at + rejected);
  - the per-item "where in the scrape" highlight uses the SAME normalizer
    (`grounding.normalize`) that produced the stored `grounding` field, so the
    two can never drift.

SQLite/threading: `Corpus` wraps ONE sqlite connection (check_same_thread=True),
and FastAPI runs sync endpoints in a threadpool -- so we do NOT share a Corpus
across requests. Each handler opens a SHORT-LIVED `with open_corpus(DB_PATH)` of
its own (opening sqlite is cheap; WAL mode makes concurrent read/write fine).

DB_PATH is a module global so tests can point it at a tmp DB
(`monkeypatch.setattr(viz.review, "DB_PATH", tmp)`); it also honours the
CORPUS_DB env var. It is read lazily inside each handler, so a monkeypatch after
import takes effect.

Run from the repo root:
    uv run uvicorn viz.review:app --host 127.0.0.1 --port 8001
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

# corpus.py + grounding.py live in src/ and use flat imports; put src/ on the path
# the same way the other entry modules do (see CLAUDE.md). Both are stdlib-only, so
# importing them keeps the "no GPU stack" promise this app relies on.
_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
_STATIC = Path(__file__).resolve().parent / "static"
sys.path.insert(0, str(_SRC))

from corpus import open_corpus  # noqa: E402
import grounding  # noqa: E402

# Overridable corpus DB path. Tests monkeypatch this; ops can set CORPUS_DB.
# Read lazily inside each handler so a monkeypatch after import takes effect.
DB_PATH = Path(os.environ.get("CORPUS_DB", str(_REPO / "data" / "corpus.sqlite")))

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


def _decision_of(trace: dict) -> str | None:
    """Map a trace's DB review state to the UI's keep|reject|None vocabulary.

    unreviewed (reviewed_at is None) -> None; reviewed & rejected -> 'reject';
    reviewed & not rejected -> 'keep'. Mirrors corpus.set_review_decision's writes.
    """
    if trace.get("reviewed_at") is None:
        return None
    return "reject" if trace.get("rejected") else "keep"


def _summary(trace: dict) -> dict:
    """A nav/list summary built purely from a trace dict (which now always carries
    grounding as a field, so there is no separate index to consult)."""
    fj = trace.get("final_json") or {}
    return {
        "trace_id": trace["trace_id"],
        "restaurant_name": trace.get("restaurant_name") or fj.get("restaurant_name") or "",
        "episode_input": trace.get("episode_input") or "",
        "dietary_restrictions": trace.get("dietary_restrictions"),
        "found": bool(trace.get("found")),
        "schema_valid": bool(trace.get("schema_valid")),
        "n_items": _n_items(fj),
        "grounding": trace.get("grounding"),
        "decision": _decision_of(trace),
    }


def _aggregate(counts: dict) -> dict:
    """The header's progress aggregate, from corpus.review_counts() -- GLOBAL over
    every trace (not the current scope). Shape is kept stable for the frontend:
    total / kept / rejected / undecided."""
    kept = int(counts.get("kept", 0) or 0)
    rejected = int(counts.get("rejected", 0) or 0)
    unreviewed = int(counts.get("unreviewed", 0) or 0)
    reviewed = int(counts.get("reviewed", 0) or 0)
    return {"total": reviewed + unreviewed, "kept": kept,
            "rejected": rejected, "undecided": unreviewed}


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
    to tell the reviewer WHICH scrape a matched item came from -- provenance the
    shared grounding scorer doesn't track, so this stays a review-local helper.
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
    """Per menu item: does its name appear in the tool text, and where.

    The match test uses `grounding.normalize` -- the SAME normalizer
    scripts/corpus/build_corpus.py feeds through `grounding.score` to compute the
    stored `grounding` field -- so a per-item "found in scrape" here can never
    disagree with the aggregate the row already carries. Each segment's whitespace
    is collapsed exactly as `grounding.score` collapses its haystack; original case
    is kept for the display window (lowercasing is length-preserving, so an offset
    into the lowercased copy maps back into the original-case one).

    For a match, `context` is a ±SIBLING_CONTEXT_CHARS window around the first
    occurrence with the matched span wrapped in _MATCH_OPEN/_MATCH_CLOSE;
    `source_hint` is the scrape URL whose result contains the match.

    Returns [{name, matched: bool, context: str|None, source_hint: str|None}].
    """
    # (collapsed original-case, collapsed lowercased, url) per segment.
    prepared = []
    for raw, url in segments:
        collapsed = re.sub(r"\s+", " ", raw)
        prepared.append((collapsed, collapsed.lower(), url))

    out: list[dict] = []
    for section in menu or []:
        if not isinstance(section, dict):
            continue
        for it in section.get("items") or []:
            name = (it.get("name") or "").strip() if isinstance(it, dict) else ""
            if not name:
                continue
            core = grounding.normalize(name)
            hit = None
            for collapsed, low, url in prepared:
                idx = low.find(core) if core else -1
                if idx >= 0:
                    hit = (collapsed, idx, url)
                    break
            if hit is None:
                out.append({"name": name, "matched": False,
                            "context": None, "source_hint": None})
                continue
            collapsed, idx, url = hit
            start = max(0, idx - SIBLING_CONTEXT_CHARS)
            end = min(len(collapsed), idx + len(core) + SIBLING_CONTEXT_CHARS)
            snippet = (collapsed[start:idx] + _MATCH_OPEN
                       + collapsed[idx:idx + len(core)] + _MATCH_CLOSE
                       + collapsed[idx + len(core):end]).strip()
            if start > 0:
                snippet = "…" + snippet
            if end < len(collapsed):
                snippet = snippet + "…"
            out.append({"name": name, "matched": True,
                        "context": snippet, "source_hint": url})
    return out


def _siblings_for(cx, trace: dict) -> list:
    """Sibling entries for a trace: the other slices of the same restaurant (free +
    dietary-conditioned, via `Corpus.siblings`), each with its found/%-grounded
    state and its rendered menu so the reviewer can compare slices and cross-check
    identity.

    Each entry carries the sibling's own stored `decision` (keep|reject|None) so
    the panel can render its state on load. Loads must happen while `cx` is open.
    """
    rid = trace.get("restaurant_id")
    tid = trace["trace_id"]
    if not rid:
        return []
    out: list = []
    for sib_id in cx.siblings(rid, exclude=tid):
        sib = cx.get_trace(sib_id)
        if sib is None:
            continue
        sib_fj = sib.get("final_json") or {}
        trimmed = _trim_menu(sib_fj.get("menu"))
        segments = _sibling_tool_segments(sib)
        out.append({
            "sibling_trace_id": sib_id,
            "restaurant_name": sib.get("restaurant_name"),
            "restrictions": sib.get("dietary_restrictions"),
            "found": bool(sib.get("found")),
            "grounding": sib.get("grounding"),
            "n_items": _n_items(sib_fj),
            "source_url": sib_fj.get("source_url"),
            "unmatched_items": sib.get("unmatched_items") or [],
            "menu": trimmed,
            # The sibling's own keep/reject/None state (shown as a badge + controls).
            "decision": _decision_of(sib),
            # Per-item: did this item's name appear in the sibling's scraped text,
            # and if so a ±context window with the match marked (the "where").
            "items": _match_menu_items(trimmed, segments),
        })
    # Found slices first, then most-suspicious (lowest grounding) among them; the
    # abstained (not-found) siblings sort to the end.
    out.sort(key=lambda s: (not s.get("found"),
                            s.get("grounding") if isinstance(s.get("grounding"), (int, float)) else 1.0))
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
    (GET /api/review/toolresult/{trace_id}?turn=&idx=).
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
    trace_id: str
    # "keep" | "reject" to set; None or "undo" to clear (un-review the trace).
    decision: str | None = None
    # Optional free-text reason, stored as reject_reason on a reject.
    reason: str | None = None


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_STATIC / "review.html")


@app.get("/api/review/traces")
def list_traces(scope: str = "notfound") -> dict:
    """Ordered summaries for the nav + a GLOBAL review-progress aggregate.

    scope=notfound (default): only found=false traces -- the review task.
    scope=all: every trace in the corpus (rejected traces included, so a review
    decision is never hidden). Sorted not-found first, then by trace_id. Each
    summary carries the trace's own %-grounded field.
    """
    scope = (scope or "notfound").lower()
    with open_corpus(DB_PATH) as cx:
        traces = list(cx.iter_traces(include_rejected=True))
        counts = cx.review_counts()

    summaries = [_summary(t) for t in traces if scope == "all" or not t.get("found")]
    # Not-found first, then trace_id (stable within each group).
    summaries.sort(key=lambda s: (s["found"], s["trace_id"]))
    return {"scope": scope, "traces": summaries, "aggregate": _aggregate(counts)}


@app.get("/api/review/trace/{trace_id}")
def get_trace(trace_id: str) -> dict:
    with open_corpus(DB_PATH) as cx:
        trace = cx.get_trace(trace_id)
        if trace is None:
            return {"ok": False, "error": f"Trace not found: {trace_id!r}"}
        siblings = _siblings_for(cx, trace)

    fj = trace.get("final_json") or {}
    resp = {
        "ok": True,
        "trace_id": trace["trace_id"],
        "restaurant_name": trace.get("restaurant_name") or fj.get("restaurant_name") or "",
        "episode_input": trace.get("episode_input") or "",
        "dietary_restrictions": trace.get("dietary_restrictions"),
        "found": bool(trace.get("found")),
        "schema_valid": bool(trace.get("schema_valid")),
        "n_items": _n_items(fj),
        # %-grounded for THIS trace: how much of its own menu is in what it scraped
        # (a stored field, computed at capture time by build_corpus).
        "grounding": trace.get("grounding"),
        "final_json": fj,
        "menu": fj.get("menu") or [],
        "notes": fj.get("notes"),
        "source_url": fj.get("source_url"),
        "queries": trace.get("queries") or [],
        "urls": trace.get("urls") or [],
        "decision": _decision_of(trace),
        "conversation": _compact_conversation(trace.get("messages") or []),
    }
    if siblings:
        resp["siblings"] = siblings
    return resp


@app.get("/api/review/toolresult/{trace_id}")
def get_tool_result(trace_id: str, turn: int = 0, idx: int = 0) -> dict:
    """Full (untruncated, capped at TOOL_RESULT_MAX_CHARS) text of one tool_result.

    Addressed by (turn, idx) into the same _walk_turns ordering the compact
    conversation exposes, so the page can expand a preview on demand while default
    payloads stay small. A bogus trace_id simply misses in the DB -> {ok: False}.
    """
    with open_corpus(DB_PATH) as cx:
        trace = cx.get_trace(trace_id)
    if trace is None:
        return {"ok": False, "error": f"Trace not found: {trace_id!r}"}
    turns = _walk_turns(trace.get("messages") or [])
    if turn < 0 or turn >= len(turns):
        return {"ok": False, "error": f"turn {turn} out of range (0..{len(turns) - 1})"}
    results = turns[turn]["tool_results"]
    if idx < 0 or idx >= len(results):
        return {"ok": False, "error": f"idx {idx} out of range (0..{len(results) - 1})"}
    raw = results[idx]["text"]
    return {
        "ok": True,
        "trace_id": trace_id,
        "turn": turn,
        "idx": idx,
        "text": raw[:TOOL_RESULT_MAX_CHARS],
        "full_len": len(raw),
        "truncated": len(raw) > TOOL_RESULT_MAX_CHARS,
    }


@app.post("/api/review/decision")
def post_decision(req: DecisionRequest) -> dict:
    """Record a keep/reject/undo decision straight into corpus.sqlite and return the
    refreshed GLOBAL aggregate. 'undo' (or null/empty) un-reviews the trace."""
    choice = req.decision
    if choice in (None, "undo", "", "null"):
        corpus_decision, new_decision = "undecided", None
    elif choice == "keep":
        corpus_decision, new_decision = "keep", "keep"
    elif choice == "reject":
        corpus_decision, new_decision = "reject", "reject"
    else:
        return {"ok": False, "error": f"Invalid decision: {choice!r} (want keep|reject|undo)"}

    with open_corpus(DB_PATH) as cx:
        if not cx.has_trace(req.trace_id):
            return {"ok": False, "error": f"Trace not found: {req.trace_id!r}"}
        cx.set_review_decision(req.trace_id, corpus_decision, req.reason)
        counts = cx.review_counts()

    return {"ok": True, "trace_id": req.trace_id, "decision": new_decision,
            "aggregate": _aggregate(counts)}
