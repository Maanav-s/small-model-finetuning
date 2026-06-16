"""The menu JSON contract — the single source of truth for the project.

The prompt describes this schema, the loop's output is validated against it, and
the Phase 3 GRPO reward will score against it. Keep the two representations in
sync: SCHEMA_SNIPPET is the human-readable block shown to the model;
MENU_SCHEMA is the machine-checkable form used by code.
"""

from __future__ import annotations

import json
import re

# Human-readable schema embedded in the system prompt (see prompts.py).
SCHEMA_SNIPPET = """\
{
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

# Machine-checkable mirror of SCHEMA_SNIPPET (for validation / reward).
MENU_SCHEMA = {
    "type": "object",
    "properties": {
        "restaurant_name": {"type": "string"},
        "cuisine": {"type": "string"},
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
    },
    "required": ["restaurant_name", "menu"],
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
