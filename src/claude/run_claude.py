"""CLI entry point for the Claude baseline (restaurant name -> menu JSON).

The Claude counterpart to run_agent.py: it parses args, picks the **same** tool
source (tools.py), runs one episode through Claude Sonnet (claude_agent.py), and
reports/validates the result against the **same** JSON contract (schema.py). No
model weights are loaded — this talks to the Anthropic API — so it runs without
a GPU and is fast to iterate on.

  uv run python src/claude/run_claude.py             # live web tools (Brave + Jina)
  uv run python src/claude/run_claude.py --offline   # offline web_search stub

Requires ANTHROPIC_API_KEY in the env (or repo-root .env); the live (default)
tool path additionally requires BRAVE_API_KEY and JINA_API_KEY.
"""

import argparse
import os
import sys
from pathlib import Path

# Shared modules (schema/prompts/tools) live in src/, the parent of this claude/
# folder; put it on the path so the flat imports below resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import anthropic  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from claude_agent import MODEL_ID, run_episode  # noqa: E402
from prompts import TEST_RESTAURANT, normalize_dietary_restrictions  # noqa: E402
from schema import extract_json  # noqa: E402
from tools import setup_tools  # noqa: E402

# Load ANTHROPIC_API_KEY / BRAVE_API_KEY / JINA_API_KEY from the repo-root .env
# regardless of cwd (this file lives in src/claude/). The search/scrape keys are
# needed for the default live tool path; --offline needs only ANTHROPIC_API_KEY.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use the deterministic local web_search stub (returns sample_menu.md) "
        "instead of the live web tools. Default is live, which requires "
        "BRAVE_API_KEY and JINA_API_KEY in the env.",
    )
    parser.add_argument(
        "--model",
        default=MODEL_ID,
        help=f"Claude model id (default: {MODEL_ID}).",
    )
    parser.add_argument(
        "--dietary",
        nargs="*",
        default=None,
        metavar="RESTRICTION",
        help="Dietary restrictions to filter the menu by, e.g. "
        '--dietary vegetarian "no nuts" (or a single comma-separated string). '
        "Omit for no filtering (the full menu).",
    )
    parser.add_argument(
        "--prompt-variant",
        choices=["teacher", "student"],
        default="teacher",
        help="System-prompt variant. 'teacher' (default) includes the "
        "source-selection guidance (prefer the restaurant's own site, avoid "
        "delivery apps); 'student' omits it. The plan is to distill teacher "
        "behavior into the student via context distillation (see CLAUDE.md).",
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
    elif parsed.get("found") is False:
        print(f"Valid JSON. MENU NOT FOUND for "
              f"{parsed.get('restaurant_name')!r}: {parsed.get('notes')!r}")
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
    tools, tool_registry, system_prompt = setup_tools(
        offline=args.offline,
        dietary_restrictions=args.dietary,
        variant=args.prompt_variant,
    )
    restaurant = TEST_RESTAURANT
    diet = normalize_dietary_restrictions(args.dietary)
    print(f"\n=== Episode ({args.model}): {restaurant} ===")
    print(f"Prompt variant: {args.prompt_variant}")
    print(f"Dietary restrictions: {', '.join(diet) if diet else '(none)'}")
    answer = run_episode(
        client, restaurant, tools, tool_registry, system_prompt, model=args.model
    )

    report(answer)


if __name__ == "__main__":
    main()
