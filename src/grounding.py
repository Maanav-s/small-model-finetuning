"""Menu-grounding: the single normalizer + scorer shared by the corpus build,
the review UI, and analysis.

Grounding = the fraction of a trace's own extracted menu item names that literally
appear in the text it scraped/searched (its OWN captured tool output).

  grounding high  -> the model really read those items off the page it scraped.
  grounding low   -> it may have INVENTED items, or extracted a DIFFERENT
                     restaurant's page (same-name, wrong-city).

In v1 this logic lived in TWO places kept in sync by hand -- `scripts/audit_grounding.py`
and `viz/review._match_menu_items`. It now lives here once; both import it, so the
aggregate score and the per-item "where in the scrape" highlight can never drift.

`build_corpus` computes grounding at capture time and stores it on the trace
(`traces.grounding` / `traces.unmatched_items`), so there is no separate post-hoc
scan (the old `audit_grounding.py` is retired).

`tool_text` is format-agnostic: it extracts tool-result text from BOTH the
Anthropic content-block shape (v1 Claude teacher) and the OpenAI chat shape
(v2 vLLM teacher / student rollouts), so grounding works regardless of which
teacher produced the trace.
"""

from __future__ import annotations

import re

# Cap the stored miss list so a wildly-hallucinated menu can't bloat a row.
UNMATCHED_CAP = 25


def normalize(s: str) -> str:
    """Canonical match form: lowercase, collapse whitespace runs to one space, strip.

    Every grounding comparison (aggregate here + the per-item highlight in the
    review UI) MUST go through this, or the two disagree.
    """
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def _extract_text_blocks(content) -> list[str]:
    """Pull plain text out of a message `content` that may be a str or a list of
    blocks. Handles both Anthropic ({type:text, text:...}) and loose shapes."""
    out: list[str] = []
    if isinstance(content, str):
        out.append(content)
    elif isinstance(content, list):
        for b in content:
            if isinstance(b, dict):
                if b.get("type") in (None, "text") and isinstance(b.get("text"), str):
                    out.append(b["text"])
                elif isinstance(b.get("content"), (str, list)):
                    out.extend(_extract_text_blocks(b["content"]))
            elif isinstance(b, str):
                out.append(b)
    return out


def tool_text(messages: list[dict]) -> str:
    """Concatenate every tool-RESULT payload (search + scrape) in a trace, lowercased.

    Recognizes tool output in two shapes:
      * Anthropic: `tool_result` blocks inside user-role message `content` lists.
      * OpenAI:    messages with `role == "tool"` (content is a string or blocks).
    """
    parts: list[str] = []
    for m in messages or []:
        role = m.get("role")
        content = m.get("content")
        # OpenAI tool-result message.
        if role == "tool":
            parts.extend(_extract_text_blocks(content))
            continue
        # Anthropic tool_result blocks (live in user-role messages).
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    parts.extend(_extract_text_blocks(block.get("content")))
    return "\n".join(parts).lower()


def item_names(menu) -> list[str]:
    """All item names in a menu (list of sections, each with `items`)."""
    names: list[str] = []
    for section in menu or []:
        for it in section.get("items", []) or []:
            n = (it.get("name") or "").strip()
            if n:
                names.append(n)
    return names


def score(names: list[str], haystack: str) -> tuple[float, list[str]]:
    """(fraction of item names found as a normalized substring of `haystack`,
    list of the names that did NOT match). `haystack` should already be lowercased
    (as `tool_text` returns it); it is normalized here for whitespace only."""
    if not names:
        return 0.0, []
    hay = re.sub(r"\s+", " ", haystack)
    misses = [n for n in names if normalize(n) not in hay]
    matched = len(names) - len(misses)
    return matched / len(names), misses


def grounding_for_trace(trace: dict) -> tuple[float | None, list[str]]:
    """Compute (grounding, unmatched_items[:cap]) for a trace dict.

    Returns (None, []) when the final menu has no items (nothing to ground) --
    e.g. a found=false abstention. Uses `final_json.menu` against the trace's own
    captured tool output.
    """
    fj = trace.get("final_json") or {}
    names = item_names(fj.get("menu"))
    if not names:
        return None, []
    frac, misses = score(names, tool_text(trace.get("messages") or []))
    return round(frac, 3), misses[:UNMATCHED_CAP]
