"""The async tool variant: TRL's GRPO loop only parallelizes COROUTINE tools.

TRL splits declared tools by `inspect.iscoroutinefunction` (grpo_trainer.py ~1819):
sync tools are invoked inline one-at-a-time, async ones are asyncio.gather'd. With sync
tools every live scrape serialized across the generation batch -- measured >40 min/step
with the GPU at 0% (see notes/experiments.md 2026-07-16). These tests pin the two
properties that fix has to keep: TRL must SEE the tools as coroutines, and the model
must see the SAME schema it was SFT'd on.
"""

import asyncio
import inspect
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tools import build_model_tools  # noqa: E402


def _fakes(delay=0.0):
    def search(q):
        time.sleep(delay)
        return f"results for {q}"

    def scrape(u, mode="direct"):
        time.sleep(delay)
        return f"page {u} ({mode})"

    return search, scrape


def test_sync_tools_are_not_coroutines():
    """Default stays sync: eval_split/agent.py/claude call the registry directly."""
    tools, _ = build_model_tools(*_fakes())
    assert [inspect.iscoroutinefunction(t) for t in tools] == [False, False]


def test_async_tools_are_coroutines_so_trl_gathers_them():
    """The whole point: TRL routes on iscoroutinefunction."""
    tools, _ = build_model_tools(*_fakes(), async_tools=True)
    assert [inspect.iscoroutinefunction(t) for t in tools] == [True, True]


def test_async_tools_render_the_identical_schema():
    """The docstring is the model-facing contract baked into training data -- the async
    wrapper must not change what apply_chat_template(tools=...) renders."""
    from transformers.utils import get_json_schema

    sync_tools, _ = build_model_tools(*_fakes())
    async_tools, _ = build_model_tools(*_fakes(), async_tools=True)
    for s, a in zip(sync_tools, async_tools):
        assert s.__name__ == a.__name__
        assert s.__doc__ == a.__doc__
        assert str(inspect.signature(s)) == str(inspect.signature(a))
        assert get_json_schema(s) == get_json_schema(a)


def test_async_tools_return_the_same_value_as_sync():
    _, sync_reg = build_model_tools(*_fakes())
    _, async_reg = build_model_tools(*_fakes(), async_tools=True)
    assert asyncio.run(async_reg["web_search"](query="q")) == sync_reg["web_search"](query="q")
    assert asyncio.run(async_reg["scrape_url"](url="u", mode="browser")) == \
        sync_reg["scrape_url"](url="u", mode="browser")


def test_async_tools_actually_run_concurrently():
    """Regression for the real bug: blocking backends must overlap under gather.
    4 x 0.3s serially is ~1.2s; gathered it should be ~0.3s."""
    _, async_reg = build_model_tools(*_fakes(delay=0.3), async_tools=True)

    async def _four():
        t0 = time.time()
        await asyncio.gather(*[async_reg["web_search"](query=f"q{i}") for i in range(4)])
        return time.time() - t0

    assert asyncio.run(_four()) < 0.9  # generous vs the ~1.2s serial floor
