"""Mark (or re-mark) the sft/grpo/eval split on corpus.sqlite -- a thin CLI over
corpus.Corpus.assign_splits.

Split assignment is DECOUPLED from harvest (harvest leaves rows unmarked by
default) so the split can be re-derived independently. This is the tool that does
it. Two modes, both random-seeded and deterministic given (seed, id-set,
fractions):

  * default (fill-NULL): assign ONLY currently-unmarked restaurants, leaving
    existing assignments untouched -- always safe.
  * --reassign: (re)assign ALL restaurants, which can MOVE a restaurant to a
    different split. Moving a restaurant that already has traces is a leakage
    hazard (an 'sft' trace resurfacing under 'eval'), so it is REFUSED unless
    --force (see notes/v2_rebuild_plan.md, Open items).

  uv run python scripts/corpus/assign_splits.py                       # fill unmarked, 50/30/20
  uv run python scripts/corpus/assign_splits.py --fractions "sft=0.6,grpo=0.2,eval=0.2"
  uv run python scripts/corpus/assign_splits.py --reassign --force    # full re-shuffle (leak-guard off)
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# Shared modules live in src/ (flat-import, script-run convention -- see CLAUDE.md).
sys.path.insert(0, str(REPO_ROOT / "src"))

from corpus import VALID_SPLITS, open_corpus  # noqa: E402

# Default split shares (must cover the valid splits and sum to 1.0). Matches the
# v2 plan's 3-way sft/grpo/eval design (notes/v2_rebuild_plan.md §2).
DEFAULT_FRACTIONS = {"sft": 0.5, "grpo": 0.3, "eval": 0.2}


def parse_fractions(text: str) -> dict[str, float]:
    """Parse "sft=0.5,grpo=0.3,eval=0.2" into a {split: fraction} dict.

    assign_splits validates the shares (known splits, sum ~1.0); this only turns
    the string into the dict it wants.
    """
    fractions: dict[str, float] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise argparse.ArgumentTypeError(
                f"bad --fractions entry {part!r}; expected split=fraction (e.g. sft=0.5)"
            )
        split, value = part.split("=", 1)
        split = split.strip()
        if split not in VALID_SPLITS:
            raise argparse.ArgumentTypeError(
                f"unknown split {split!r} in --fractions (valid: {', '.join(VALID_SPLITS)})"
            )
        try:
            fractions[split] = float(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"non-numeric fraction {value!r} for {split!r}")
    if not fractions:
        raise argparse.ArgumentTypeError("--fractions parsed to nothing")
    return fractions


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--db", type=Path, default=REPO_ROOT / "data" / "corpus.sqlite",
                        help="corpus.sqlite path (default data/corpus.sqlite)")
    parser.add_argument("--seed", type=int, default=42,
                        help="assignment seed (default 42; keep fixed for reproducibility)")
    parser.add_argument("--fractions", type=parse_fractions, default=None,
                        help='per-split shares as "sft=0.5,grpo=0.3,eval=0.2" '
                             "(default 50/30/20); must sum to 1.0")
    parser.add_argument("--reassign", action="store_true",
                        help="(re)assign ALL restaurants, not just unmarked ones (can MOVE rows)")
    parser.add_argument("--force", action="store_true",
                        help="with --reassign, allow moving restaurants that already have "
                             "traces (leakage risk -- off by default)")
    return parser.parse_args()


def _print_counts(label: str, counts: dict[str, int]) -> None:
    total = sum(counts.values())
    parts = "  ".join(f"{s}={counts.get(s, 0)}" for s in (*VALID_SPLITS, "unmarked"))
    print(f"{label:<7} {parts}  (total {total})")


def main():
    args = parse_args()
    fractions = args.fractions or dict(DEFAULT_FRACTIONS)

    with open_corpus(args.db, create=False) as cx:
        before = cx.count_by_split()
        _print_counts("before", before)
        if before["unmarked"] == 0 and not args.reassign:
            print("nothing to do: no unmarked restaurants (pass --reassign to re-shuffle)")
            return
        try:
            assigned = cx.assign_splits(
                seed=args.seed, fractions=fractions,
                reassign=args.reassign, force=args.force,
            )
        except ValueError as exc:
            sys.exit(f"assign_splits refused: {exc}")
        after = cx.count_by_split()
        _print_counts("after", after)
        pool = "all restaurants" if args.reassign else "unmarked restaurants"
        print(f"assigned {sum(assigned.values())} {pool} "
              f"(seed {args.seed}, fractions {fractions}): "
              + ", ".join(f"{s}={assigned.get(s, 0)}" for s in VALID_SPLITS if s in fractions))


if __name__ == "__main__":
    main()
