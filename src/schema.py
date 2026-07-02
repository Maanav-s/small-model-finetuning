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
        {"name": "string", "description": "string", "price": number or null}
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
                                "description": {"type": "string"},
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
    """Best-effort: strip markdown fences and parse the answer as JSON.

    Returns (obj, None) on success or (None, error_message) on failure.
    """
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(text), None
    except json.JSONDecodeError as e:
        return None, str(e)
