"""Smoke-test a served vLLM teacher end-to-end: one restaurant -> validated menu JSON.

Exercises the SAME path build_corpus.py --teacher vllm uses (openai_agent.run_episode
over an OpenAI-compatible vLLM server, the shared setup_tools + schema.MENU_SCHEMA), so
a PASS here means the served endpoint is ready to drive a corpus build. The tools run
LOCALLY in this process (Brave web_search + local Playwright scrape), so run it where the
scrape should egress from -- typically ON the pod, co-located with the vLLM server -- with
BRAVE_API_KEY set (repo-root .env is auto-loaded).

  uv run python scripts/infra/smoke_teacher.py --base-url http://localhost:8000/v1 --model teacher
  uv run python scripts/infra/smoke_teacher.py --model teacher --restaurant "Canlis, Seattle"
  uv run python scripts/infra/smoke_teacher.py --model teacher --dietary vegan   # conditioned episode

Exit status is 0 on PASS (JSON extracted + schema-valid), 1 on FAIL -- so it doubles
as a CI-style gate before spending on a full build.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# Shared modules in src/; the teacher loop in src/serving (flat-import, script-run).
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "serving"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")  # BRAVE_API_KEY for the live tools

import jsonschema  # noqa: E402
from openai_agent import build_client, run_episode  # noqa: E402
from prompts import TEST_RESTAURANT  # noqa: E402
from schema import MENU_SCHEMA, extract_json  # noqa: E402
from tools import setup_tools  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://localhost:8000/v1",
                   help="OpenAI-compatible base URL of the vLLM server (default localhost:8000)")
    p.add_argument("--model", default="teacher",
                   help="served model name (vllm serve --served-model-name; default 'teacher')")
    p.add_argument("--restaurant", default=TEST_RESTAURANT, help=f"episode input (default {TEST_RESTAURANT!r})")
    p.add_argument("--dietary", default=None, help="optional dietary restriction to condition on (e.g. vegan)")
    args = p.parse_args()

    restrictions = [args.dietary] if args.dietary else None
    tools, registry, system_prompt = setup_tools(restrictions, "teacher", None)
    client = build_client(args.base_url)

    print(f"[smoke] {args.model} @ {args.base_url}  input={args.restaurant!r}"
          + (f"  dietary={args.dietary}" if args.dietary else ""))
    final_text, messages = run_episode(client, args.model, args.restaurant, tools, registry, system_prompt)

    tool_turns = sum(1 for m in messages if m.get("role") == "assistant" and m.get("tool_calls"))
    data, err = extract_json(final_text)
    if data is None:
        print(f"[FAIL] no JSON extracted ({err}); final text was {len(final_text)} chars:\n{final_text[:600]}")
        sys.exit(1)
    try:
        jsonschema.validate(data, MENU_SCHEMA)
    except jsonschema.ValidationError as e:
        print(f"[FAIL] JSON does not satisfy MENU_SCHEMA: {e.message}")
        sys.exit(1)

    items = sum(len(sec.get("items", [])) for sec in data.get("menu", []))
    print(f"[PASS] tool-call turns={tool_turns}  found={data.get('found')}  "
          f"sections={len(data.get('menu', []))}  items={items}  "
          f"name={data.get('restaurant_name')!r}  source={data.get('source_url')!r}")


if __name__ == "__main__":
    main()
