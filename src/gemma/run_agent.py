"""CLI entry point for the Phase 1 agent loop (restaurant name -> menu JSON).

This is a thin driver: it parses args, loads the model (model.py), picks a tool
source (tools.py), runs one episode (agent.py), and reports/validates the result
(schema.py). The reusable pieces live in those modules so the SFT/eval scripts
and a REPL can call them without going through this CLI.

  uv run python src/gemma/run_agent.py             # bf16, live web tools (Brave + Jina)
  uv run python src/gemma/run_agent.py --offline   # offline web_search stub
  uv run python src/gemma/run_agent.py --quantize  # 4-bit (low-VRAM / fast load)
"""

import argparse
import sys
from pathlib import Path

# Shared modules (schema/prompts/tools) live in src/, the parent of this gemma/
# folder; put it on the path so the flat imports below resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from agent import run_episode  # noqa: E402
from model import load_model  # noqa: E402
from prompts import TEST_RESTAURANT, normalize_dietary_restrictions  # noqa: E402
from schema import extract_json  # noqa: E402
from tools import setup_tools  # noqa: E402

# Load BRAVE_API_KEY / JINA_API_KEY (and anything else) from the repo-root .env,
# regardless of the current working directory (this file lives in src/gemma/).
load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Load the model in 4-bit (nf4). Off by default; use on low-VRAM GPUs.",
    )
    parser.add_argument(
        "--attn",
        choices=["sdpa", "eager"],
        default="sdpa",
        help="Attention kernel. Default 'sdpa' (torch built-in, no extra dep). "
        "flash_attention_2 is NOT an option: Gemma 4 E4B's global-attention layers "
        "use head_dim=512, above FlashAttention-2's hard cap of 256 (FA3/FA4 would "
        "fit but need Hopper GPUs this box lacks). 'eager' is the slow reference path.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use the deterministic local web_search stub (returns sample_menu.md) "
        "instead of the live web tools. Default is live, which requires "
        "BRAVE_API_KEY and JINA_API_KEY in the env.",
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
    """Print the raw answer and a quick JSON/schema sanity check."""
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
    model, tokenizer = load_model(quantize=args.quantize, attn=args.attn)
    tools, tool_registry, system_prompt = setup_tools(
        offline=args.offline,
        dietary_restrictions=args.dietary,
        variant=args.prompt_variant,
    )
    restaurant = TEST_RESTAURANT
    diet = normalize_dietary_restrictions(args.dietary)
    print(f"\n=== Episode: {restaurant} ===")
    print(f"Prompt variant: {args.prompt_variant}")
    print(f"Dietary restrictions: {', '.join(diet) if diet else '(none)'}")
    answer = run_episode(
        model, tokenizer, restaurant, tools, tool_registry, system_prompt
    )

    report(answer)


if __name__ == "__main__":
    main()
