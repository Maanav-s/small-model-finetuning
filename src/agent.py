"""Phase 1 building blocks: the system prompt and tool definitions for the
restaurant-menu extraction agent.

This module deliberately stops short of the generate/execute loop -it defines
*what the model sees* (system prompt + tool schema) and how to assemble the
initial message list. Run `uv run python agent.py` to print the fully rendered
prompt the model will receive.
"""

from __future__ import annotations

from pathlib import Path

# Single source of truth for the model id (imported by run_agent.py / scratch tools).
# E4B is the target model: it fits comfortably in bf16 (~9 GB) on a 23 GB card
# and is far better at tool calling than E2B (which would skip the web_search
# call and answer from memory). E2B was only ever a stopgap for the old 6 GB
# RTX 4050; pass --quantize in run_agent.py if you ever run this on a small card.
MODEL_ID = "google/gemma-4-E4B-it"

# ---------------------------------------------------------------------------
# System prompt
#
# Encodes the task and the exact output contract. The JSON schema here is the
# single source of truth for what the model must emit (mirrors project_plan.md).
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a restaurant menu extraction assistant.

You have NO prior knowledge of any restaurant's menu. You MUST call the \
`web_search` tool at least once and base your answer ONLY on what it returns - \
never answer from memory and never invent items. Return the menu as a single JSON \
object.

Output rules:
- Your final reply must be ONLY the JSON object - no prose, no markdown fences.
- The JSON must match this schema exactly:

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
}

- `price` must be a number (e.g. 12.5) or null - never a string like "$12.50".
- Use null for fields you cannot determine. Do not invent menu items that are \
not supported by the search results.
"""


# ---------------------------------------------------------------------------
# Tools
#
# Defined as plain Python functions. `apply_chat_template(tools=[...])` reads the
# signature + docstring (Google style) and converts it to the schema Gemma wants.
# Keep the docstring accurate -it becomes the tool description the model sees.
# ---------------------------------------------------------------------------
_SAMPLE_MENU = Path(__file__).with_name("sample_menu.md")


def web_search(query: str) -> str:
    """Search the web for a restaurant's menu information.

    Args:
        query: Search query, e.g. the restaurant name plus its city and the
            word "menu".
    """
    # TEMP STUB: ignores the query and returns a fixed sample menu, so the
    # agentic loop can be developed before Firecrawl is wired in. Replace this
    # body with a real Firecrawl search/scrape later (keep the signature).
    return _SAMPLE_MENU.read_text(encoding="utf-8")


# Passed to apply_chat_template(tools=...). The loop will dispatch parsed calls
# through TOOL_REGISTRY by name.
TOOLS = [web_search]
TOOL_REGISTRY = {fn.__name__: fn for fn in TOOLS}


def build_messages(restaurant_name: str) -> list[dict]:
    """Assemble the initial chat history for one extraction episode."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": restaurant_name},
    ]


if __name__ == "__main__":
    # Render-only demo: confirms the system prompt and tool schema coexist in
    # the prompt the model receives. Tokenizer-only, so no GPU / weights needed.
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    rendered = tok.apply_chat_template(
        build_messages("Joe's Pizza, New York"),
        tools=TOOLS,
        add_generation_prompt=True,
        tokenize=False,
    )
    print(rendered)
