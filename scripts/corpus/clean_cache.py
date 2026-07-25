"""Slim, reclassify, and clip a cache.sqlite in place, then reclaim the disk space.

Three passes, all retro-applying what the live tool path already does at the
source (backends.build_scrape slims; cache._set bounds storage), to a cache
captured before those rules -- so the stored file matches what the agent sees
(minus the read-time MAX_TOOL_CHARS cap) and shrinks:

  1. SLIM (backends._slim_scrape on every scrape row): drops base64 data: URIs,
     markdown images, empty links/bullets, dead hrefs, and collapses blank runs.
     The base64 blobs dominate a pre-scrubbing cache -- a single inline image
     measured 397K chars -- so this is the big reclaim. Skip with --no-slim.
  2. RECLASSIFY (cache.scrape_status on the now-slimmed text): slim_rows rewrites
     the response only, so a row classified on its RAW text keeps that verdict --
     e.g. a junk page whose raw markdown cleared MIN_CONTENT_CHARS classified 'ok',
     but slims to nothing and should be 'empty'. A stale 'ok' is a permanent hit
     serving junk under miss_policy="live". Skip with --no-reclassify.
  3. CLIP (cache.MAX_STORED_CHARS, the storage rule src/cache.py enforces on every
     write): clip any STILL-over-long response (a genuine multi-million-char page,
     e.g. a huge PDF-to-markdown) to its first --max-chars characters.

Then VACUUM to shrink the file on disk. Non-destructive to the CACHE'S PURPOSE: the
slim only removes noise the backend now strips on every fetch, and the clip bound
sits far above MAX_TOOL_CHARS, so an agent replaying a cleaned row sees
byte-identical content to what it saw uncached.

  uv run python scripts/corpus/clean_cache.py                       # data/cache.sqlite: slim + clip + vacuum
  uv run python scripts/corpus/clean_cache.py --cache-path /workspace/cache.sqlite
  uv run python scripts/corpus/clean_cache.py --dry-run             # report only, no mutation
  uv run python scripts/corpus/clean_cache.py --no-slim --max-chars 200000 --no-vacuum

Run it ON the pod (where the big cache lives) before syncing that cache to S3, or
locally against a pulled copy. Idempotent: a second run finds nothing to change.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# Shared modules live in src/ (flat-import, script-run convention -- see CLAUDE.md).
sys.path.insert(0, str(REPO_ROOT / "src"))

from backends import _slim_scrape  # noqa: E402  -- the exact source-side slim, baked into storage
from cache import MAX_STORED_CHARS, Cache, scrape_status  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cache-path", type=Path, default=REPO_ROOT / "data" / "cache.sqlite",
                        help="cache.sqlite to clean (default data/cache.sqlite)")
    parser.add_argument("--no-slim", action="store_true",
                        help="skip the slim pass (backends._slim_scrape on scrape rows); "
                             "clip-only, the pre-slim behaviour")
    parser.add_argument("--no-reclassify", action="store_true",
                        help="skip the status-reclassify pass (cache.scrape_status on the "
                             "slimmed scrape rows)")
    parser.add_argument("--max-chars", type=int, default=MAX_STORED_CHARS,
                        help=f"clip any stored response STILL longer than this after slimming "
                             f"(default {MAX_STORED_CHARS}, = cache.MAX_STORED_CHARS)")
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
    mutated = False
    try:
        print(f"{args.cache_path}: {size_before / 1e6:.1f} MB on disk")

        # Pass 1: slim scrape rows (base64 URIs, images, empty bullets, blank runs).
        if not args.no_slim:
            s = cache.slim_rows(_slim_scrape, dry_run=args.dry_run)
            verb = "would slim" if args.dry_run else "slimmed"
            print(f"  slim: {verb} {s['rows_changed']}/{s['rows_scanned']} scrape row(s), "
                  f"removing {s['chars_removed']:,} chars (largest response {s['largest_before']} chars)")
            mutated = mutated or s["applied"]

        # Pass 2: recompute status from the (now slimmed) stored text -- slim_rows
        # rewrites the response only, so a raw-classified status is stale forever
        # without this (e.g. junk that classified 'ok' but slims to 'empty').
        if not args.no_reclassify:
            r = cache.reclassify(scrape_status, dry_run=args.dry_run)
            verb = "would reclassify" if args.dry_run else "reclassified"
            trans = ", ".join(f"{k}: {v}" for k, v in sorted(r["transitions"].items())) or "none"
            print(f"  reclassify: {verb} {r['rows_changed']}/{r['rows_scanned']} scrape row(s) "
                  f"({trans})")
            mutated = mutated or r["applied"]

        # Pass 3: clip anything STILL over the storage bound after slimming.
        c = cache.clip_oversized(args.max_chars, dry_run=args.dry_run)
        verb = "would clip" if args.dry_run else "clipped"
        print(f"  clip: {verb} {c['rows']} row(s) over {args.max_chars} chars, "
              f"removing {c['chars_removed']:,} chars (largest {c['largest_before']} chars)")
        mutated = mutated or c["applied"]

        if args.dry_run:
            print("  (dry-run: nothing written; re-run without --dry-run to apply)")
            return
        if mutated and not args.no_vacuum:
            print("  vacuuming to reclaim freed pages on disk...")
            cache.vacuum()
    finally:
        cache.close()

    size_after = args.cache_path.stat().st_size
    print(f"  size: {size_before / 1e6:.1f} MB -> {size_after / 1e6:.1f} MB "
          f"(reclaimed {(size_before - size_after) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
