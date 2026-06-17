"""System prompts for the menu-extraction agent.

The output contract is imported from schema.py (SCHEMA_SNIPPET) so the prompt the
model sees and the schema the code validates against never drift apart.
"""

from __future__ import annotations

from schema import SCHEMA_SNIPPET

# Restaurant used as the episode input while testing the loop end-to-end. Both
# runners (gemma/run_agent.py, claude/run_claude.py) read this. Temporary: real
# eval will iterate over a dataset of restaurants rather than a single name.
TEST_RESTAURANT = "Thai Trio, Sammamish, WA"

SYSTEM_PROMPT = f"""\
You are a restaurant menu extraction assistant.

You have NO prior knowledge of any restaurant's menu. You MUST call the \
`web_search` tool at least once and base your answer ONLY on what it returns - \
never answer from memory and never invent items. Return the menu as a single JSON \
object.

Output rules:
- Your final reply must be ONLY the JSON object - no prose, no markdown fences.
- The JSON must match this schema exactly:

{SCHEMA_SNIPPET}

- Include ONLY main course menu items (entrees / mains). Do NOT include \
appetizers, starters, sides, desserts, or drinks/beverages.
- `price` must be a number (e.g. 12.5) or null - never a string like "$12.50".
- Use null for fields you cannot determine. Do not invent menu items that are \
not supported by the search results.
"""

# MCP path: same contract, but names the Firecrawl tools instead of web_search.
MCP_SYSTEM_PROMPT = SYSTEM_PROMPT.replace(
    "You MUST call the `web_search` tool at least once",
    "You MUST call the `firecrawl_search` tool at least once (and may then call "
    "`firecrawl_scrape` on a promising result URL to read the full menu page)",
) + """\

Tool-use rules:
- When you call `firecrawl_scrape`, request the page as plain markdown
  (`formats: ["markdown"]`). Read the returned text yourself and build the menu
  JSON in YOUR final answer.
- Do NOT use Firecrawl's structured-extraction (`json` format / `jsonOptions` /
  `schema`) and do NOT pass a JSON schema as a tool argument. You are the one
  who produces the JSON, not the tool.
"""
