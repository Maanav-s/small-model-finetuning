"""The menu JSON contract — the single source of truth for the project.

The prompt describes this schema, the loop's output is validated against it, and
the Phase 3 GRPO reward will score against it. Keep the two representations in
sync: SCHEMA_SNIPPET is the human-readable block shown to the model;
MENU_SCHEMA is the machine-checkable form used by code.
"""

from __future__ import annotations

import json
import re

# Sentinel for an item whose price could not be found. `price` is a number when
# known and PRICE_UNKNOWN (null) when the menu lists no price / none was found --
# the model is told never to guess a price, so null is unambiguous "unknown".
PRICE_UNKNOWN = None

# Human-readable schema embedded in the system prompt (see prompts.py). `found`
# is true for a normal result; the NOT_FOUND_SNIPPET below is the shape to return
# when no menu could be found at all.
SCHEMA_SNIPPET = """\
{
  "found": true,
  "restaurant_name": "string",
  "cuisine": "string",
  "menu": [
    {
      "section": "string",
      "items": [
        {"name": "string", "description": "string or null", "price": number or null}
      ]
    }
  ],
  "source_url": "string or null"
}"""

# The shape to return when the restaurant's menu cannot be found at all (no search
# result / page has it). `found` is false, `menu` is empty, and `notes` says why.
# This is a distinct, machine-detectable outcome from "found a menu but nothing
# survived the dietary filter" (that stays found=true with an empty menu).
NOT_FOUND_SNIPPET = """\
{
  "found": false,
  "restaurant_name": "string",
  "cuisine": "string or null",
  "menu": [],
  "source_url": "string or null",
  "notes": "short string explaining why the menu could not be found"
}"""

# Machine-checkable mirror of SCHEMA_SNIPPET (for validation / reward). Covers
# both the normal (found=true) and not-found (found=false) shapes.
MENU_SCHEMA = {
    "type": "object",
    "properties": {
        "found": {"type": "boolean"},
        "restaurant_name": {"type": "string"},
        "cuisine": {"type": ["string", "null"]},
        "menu": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "section": {"type": "string"},
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                # null == no description on the menu (the system
                                # prompt says "use null for fields you cannot
                                # determine", so the validator must accept it).
                                "description": {"type": ["string", "null"]},
                                # null == price could not be found (PRICE_UNKNOWN).
                                "price": {"type": ["number", "null"]},
                            },
                            "required": ["name"],
                        },
                    },
                },
                "required": ["section", "items"],
            },
        },
        "source_url": {"type": ["string", "null"]},
        "notes": {"type": ["string", "null"]},
    },
    "required": ["found", "restaurant_name", "menu"],
}


def extract_json(text: str):
    """Best-effort: strip markdown fences / surrounding prose and parse as JSON.

    Returns (obj, None) on success or (None, error_message) on failure.

    The model is told to reply with the raw JSON object only, but ~5% of teacher
    episodes still wrap it in a leading narration ("Now I'll compile the JSON...")
    or trailing commentary while the JSON object itself is complete and valid.
    Discarding those wastes a paid episode with a good menu inside, so we recover
    it: on a direct-parse failure, find the first `{` and decode a single JSON
    value from there (raw_decode stops at the end of the first complete object, so
    braces inside strings are handled and trailing prose is ignored). This parser
    is shared by the corpus builder's schema_valid gate, the eval harness, and the
    future GRPO reward, so robustness here directly affects yield and reward
    correctness. (For SFT the *raw* preamble still lives in the trace's final turn;
    WS-I renders from a cleaned final answer -- see notes/phase2_plan.md.)
    """
    stripped = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(stripped), None
    except json.JSONDecodeError as err:
        start = stripped.find("{")
        if start != -1:  # decode one object from the first '{' (drops leading
            try:        # narration and any trailing commentary; None if truncated)
                return json.JSONDecoder().raw_decode(stripped[start:])[0], None
            except json.JSONDecodeError:
                pass
        return None, str(err)
