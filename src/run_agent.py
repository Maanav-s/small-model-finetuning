"""CLI entry point for the Phase 1 agent loop (restaurant name -> menu JSON).

This is a thin driver: it parses args, loads the model (model.py), picks a tool
source (tools.py), runs one episode (agent.py), and reports/validates the result
(schema.py). The reusable pieces live in those modules so the SFT/eval scripts
and a REPL can call them without going through this CLI.

  uv run python src/run_agent.py             # bf16, offline stub tools
  uv run python src/run_agent.py --mcp       # Firecrawl MCP tools
  uv run python src/run_agent.py --quantize  # 4-bit (low-VRAM / fast load)
"""

import argparse
from pathlib import Path

from dotenv import load_dotenv

from agent import run_episode
from model import load_model
from schema import extract_json
from tools import setup_tools

# Load FIRECRAWL_API_KEY (and anything else) from the repo-root .env, regardless
# of the current working directory (this file lives in src/).
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


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
        "--mcp",
        action="store_true",
        help="Source tools from the Firecrawl MCP server (npx -y firecrawl-mcp) "
        "instead of the local web_search stub. Requires FIRECRAWL_API_KEY in the env "
        "and Node/npx on PATH.",
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
    else:
        sections = parsed.get("menu", [])
        n_items = sum(len(s.get("items", [])) for s in sections)
        print(f"Valid JSON. restaurant_name={parsed.get('restaurant_name')!r}, "
              f"cuisine={parsed.get('cuisine')!r}, "
              f"{len(sections)} sections, {n_items} items")


def main():
    args = parse_args()
    model, tokenizer = load_model(quantize=args.quantize, attn=args.attn)
    tools, tool_registry, system_prompt, mcp_client = setup_tools(args.mcp)
    try:
        restaurant = "Pagliacci Pizza, Seattle"
        print(f"\n=== Episode: {restaurant} ===")
        answer = run_episode(
            model, tokenizer, restaurant, tools, tool_registry, system_prompt
        )
    finally:
        if mcp_client is not None:
            mcp_client.close()

    report(answer)


if __name__ == "__main__":
    main()
