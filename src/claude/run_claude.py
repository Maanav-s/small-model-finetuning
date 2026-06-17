"""CLI entry point for the Claude baseline (restaurant name -> menu JSON).

The Claude counterpart to run_agent.py: it parses args, picks the **same** tool
source (tools.py), runs one episode through Claude Sonnet (claude_agent.py), and
reports/validates the result against the **same** JSON contract (schema.py). No
model weights are loaded — this talks to the Anthropic API — so it runs without
a GPU and is fast to iterate on.

  uv run python src/claude/run_claude.py             # offline web_search stub tools
  uv run python src/claude/run_claude.py --mcp       # Firecrawl MCP tools

Requires ANTHROPIC_API_KEY in the env (or repo-root .env); --mcp additionally
requires FIRECRAWL_API_KEY and Node/npx on PATH.
"""

import argparse
import os
import sys
from pathlib import Path

# Shared modules (schema/prompts/tools/mcp_client) live in src/, the parent of
# this claude/ folder; put it on the path so the flat imports below resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import anthropic  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from claude_agent import MODEL_ID, run_episode  # noqa: E402
from prompts import TEST_RESTAURANT  # noqa: E402
from schema import extract_json  # noqa: E402
from tools import setup_tools  # noqa: E402

# Load ANTHROPIC_API_KEY / FIRECRAWL_API_KEY from the repo-root .env regardless
# of cwd (this file lives in src/claude/).
load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mcp",
        action="store_true",
        help="Source tools from the Firecrawl MCP server (npx -y firecrawl-mcp) "
        "instead of the local web_search stub. Requires FIRECRAWL_API_KEY in the env "
        "and Node/npx on PATH.",
    )
    parser.add_argument(
        "--model",
        default=MODEL_ID,
        help=f"Claude model id (default: {MODEL_ID}).",
    )
    return parser.parse_args()


def report(answer: str) -> None:
    """Print the raw answer and a quick JSON/schema sanity check (cf. run_agent.py)."""
    print("\n=== RAW FINAL ANSWER ===")
    print(answer)

    parsed, err = extract_json(answer)
    print("\n=== SCHEMA CHECK ===")
    if parsed is None:
        print(f"INVALID JSON: {err}")
    else:
        sections = parsed.get("menu", [])
        n_items = sum(len(s.get("items", [])) for s in sections)
        print(f"Valid JSON. restaurant_name={parsed.get('restaurant_name')!r}, "
              f"cuisine={parsed.get('cuisine')!r}, "
              f"{len(sections)} sections, {n_items} items")


def main():
    args = parse_args()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "ANTHROPIC_API_KEY is required (set it in the environment or repo-root "
            ".env; get one at https://console.anthropic.com)."
        )

    client = anthropic.Anthropic()
    tools, tool_registry, system_prompt, mcp_client = setup_tools(args.mcp)
    try:
        restaurant = TEST_RESTAURANT
        print(f"\n=== Episode ({args.model}): {restaurant} ===")
        answer = run_episode(
            client, restaurant, tools, tool_registry, system_prompt, model=args.model
        )
    finally:
        if mcp_client is not None:
            mcp_client.close()

    report(answer)


if __name__ == "__main__":
    main()
