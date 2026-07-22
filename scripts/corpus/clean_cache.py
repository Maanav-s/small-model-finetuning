"""Clip over-long responses out of a cache.sqlite and reclaim the disk space.

Retro-applies cache.MAX_STORED_CHARS (the general storage rule that src/cache.py
now enforces on every write) to a cache built BEFORE that rule. A single monster
scrape -- a multi-million-char PDF rendered to markdown, one was 14M chars -- can
bloat the shared cache.sqlite far out of proportion to its usefulness (the read
path never surfaces more than MAX_TOOL_CHARS anyway). This walks the DB, clips any
response longer than --max-chars to its first --max-chars characters, then VACUUMs
to shrink the file on disk.

Non-destructive to the CACHE'S PURPOSE: the read-time MAX_TOOL_CHARS cap (tools.py)
is far below the clip bound, so an agent replaying a clipped row sees byte-identical
content to what it saw uncached. Only the unused tail past --max-chars is dropped.

  uv run python scripts/corpus/clean_cache.py                       # data/cache.sqlite, clip>400k + vacuum
  uv run python scripts/corpus/clean_cache.py --cache-path /workspace/cache.sqlite
  uv run python scripts/corpus/clean_cache.py --dry-run             # report only, no mutation
  uv run python scripts/corpus/clean_cache.py --max-chars 200000 --no-vacuum

Run it ON the pod (where the big cache lives) before syncing that cache to S3, or
locally against a pulled copy. Idempotent: a second run finds nothing to clip.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# Shared modules live in src/ (flat-import, script-run convention -- see CLAUDE.md).
sys.path.insert(0, str(REPO_ROOT / "src"))

from cache import MAX_STORED_CHARS, Cache  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cache-path", type=Path, default=REPO_ROOT / "data" / "cache.sqlite",
                        help="cache.sqlite to clean (default data/cache.sqlite)")
    parser.add_argument("--max-chars", type=int, default=MAX_STORED_CHARS,
                        help=f"clip any stored response longer than this (default "
                             f"{MAX_STORED_CHARS}, = cache.MAX_STORED_CHARS)")
    parser.add_argument("--no-vacuum", action="store_true",
                        help="skip the VACUUM after clipping (the clip alone only frees "
                             "pages inside the file; without VACUUM the file does not shrink "
                             "on disk). VACUUM needs free disk ~= the db size for its temp copy.")
    parser.add_argument("--dry-run", action="store_true",
                        help="report how many rows WOULD be clipped without mutating anything")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.cache_path.is_file():
        sys.exit(f"no cache at {args.cache_path}")

    size_before = args.cache_path.stat().st_size
    cache = Cache(str(args.cache_path), miss_policy="live")
    try:
        report = cache.clip_oversized(args.max_chars, dry_run=args.dry_run)
        verb = "would clip" if args.dry_run else "clipped"
        print(f"{args.cache_path}: {size_before / 1e6:.1f} MB on disk")
        print(f"  rows over {args.max_chars} chars: {report['rows']}  "
              f"(largest response {report['largest_before']} chars)")
        print(f"  {verb} {report['rows']} row(s), removing {report['chars_removed']:,} chars of text")
        if args.dry_run:
            print("  (dry-run: nothing written; re-run without --dry-run to apply)")
            return
        if report["rows"] and not args.no_vacuum:
            print("  vacuuming to reclaim freed pages on disk...")
            cache.vacuum()
    finally:
        cache.close()

    size_after = args.cache_path.stat().st_size
    print(f"  size: {size_before / 1e6:.1f} MB -> {size_after / 1e6:.1f} MB "
          f"(reclaimed {(size_before - size_after) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
