"""Harvest a stratified restaurant corpus into corpus.sqlite (v2).

Adapts the v1 scripts/harvest_restaurants.py to the v2 store: instead of writing
a loose data/restaurants.jsonl + data/splits.json, it upserts LEAN rows
(name, city, source, is_chain) straight into corpus.sqlite via
corpus.Corpus.upsert_restaurants. The v1 extras (lat/lng, region, country,
price_tier, cuisine) are NOT persisted -- lat/lng and cuisine are used only
transiently here (dedup + stratified sampling), per notes/v2_rebuild_plan.md §2.

Split marking is DECOUPLED from harvest (plan §2): harvest leaves rows unmarked
by default (--assign-split none). Pass --assign-split random to fill the split in
the same run (fills only currently-unmarked rows, seeded), or run the dedicated
scripts/corpus/assign_splits.py later.

Sourcing is pluggable behind a clean fetch seam (`SOURCES`), but only OSM/Overpass
is implemented today (no paid API key exists). The `source` column records
provenance so a later Yelp/Foursquare harvester slots in without a schema change.

  uv run python scripts/corpus/harvest.py                        # full default OSM harvest
  uv run python scripts/corpus/harvest.py --regions seattle --target 50
  uv run python scripts/corpus/harvest.py --assign-split random  # harvest + seed the split
"""

import argparse
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import requests
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
# Shared modules live in src/ (flat-import, script-run convention -- see CLAUDE.md).
sys.path.insert(0, str(REPO_ROOT / "src"))

# Load GOOGLE_PLACES_API_KEY (for --enrich-places) from the repo-root .env.
load_dotenv(REPO_ROOT / ".env")

from corpus import VALID_SPLITS, open_corpus, restaurant_id_for  # noqa: E402

DEFAULT_ENDPOINT = "https://overpass-api.de/api/interpreter"
OVERPASS_TIMEOUT_S = 180          # server-side [timeout:...] in the query
HTTP_TIMEOUT_S = OVERPASS_TIMEOUT_S + 30
SLEEP_BETWEEN_REGIONS_S = 3.0     # politeness gap between region queries
MAX_ATTEMPTS = 4                  # per region: 1 try + 3 retries
BACKOFF_BASE_S = 10.0             # 10s, 30s, 90s exponential backoff
TOP_CUISINES = 15                 # cuisines kept as their own stratum; rest -> "other"
MIN_INDEP_FRAC = 0.70             # sampling aims for >= this fraction independents
CHAIN_MIN_LOCATIONS = 3           # name-frequency heuristic threshold for is_chain

# Default split shares when --assign-split random is passed (plan §2's 3-way split).
DEFAULT_FRACTIONS = {"sft": 0.5, "grpo": 0.3, "eval": 0.2}

# Metro-scale bounding boxes: (south, west, north, east). Deliberately modest
# (city cores, not whole states) so a single Overpass query per region stays
# tractable. `region` is the state/province/country-subdivision fallback used
# when the element has no addr:state/addr:province tag.
REGIONS = {
    "seattle":      {"city": "Seattle",      "region": "Washington",       "country": "US", "bbox": (47.48, -122.44, 47.74, -122.22)},
    "portland":     {"city": "Portland",     "region": "Oregon",           "country": "US", "bbox": (45.43, -122.79, 45.60, -122.52)},
    "sf_bay_area":  {"city": "San Francisco", "region": "California",      "country": "US", "bbox": (37.70, -122.52, 37.83, -122.35)},
    "chicago":      {"city": "Chicago",      "region": "Illinois",         "country": "US", "bbox": (41.75, -87.75, 42.00, -87.55)},
    "austin":       {"city": "Austin",       "region": "Texas",            "country": "US", "bbox": (30.15, -97.85, 30.45, -97.65)},
    "philadelphia": {"city": "Philadelphia", "region": "Pennsylvania",     "country": "US", "bbox": (39.87, -75.28, 40.05, -75.06)},
    "atlanta":      {"city": "Atlanta",      "region": "Georgia",          "country": "US", "bbox": (33.65, -84.55, 33.89, -84.28)},
    "toronto":      {"city": "Toronto",      "region": "Ontario",          "country": "CA", "bbox": (43.58, -79.55, 43.76, -79.28)},
    "vancouver":    {"city": "Vancouver",    "region": "British Columbia", "country": "CA", "bbox": (49.20, -123.23, 49.32, -123.02)},
    "london":       {"city": "London",       "region": "England",          "country": "UK", "bbox": (51.44, -0.24, 51.57, 0.02)},
    "manchester":   {"city": "Manchester",   "region": "England",          "country": "UK", "bbox": (53.40, -2.32, 53.52, -2.15)},
    "sydney":       {"city": "Sydney",       "region": "New South Wales",  "country": "AU", "bbox": (-33.95, 151.10, -33.80, 151.29)},
    "melbourne":    {"city": "Melbourne",    "region": "Victoria",         "country": "AU", "bbox": (-37.88, 144.90, -37.75, 145.05)},
}

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)

# addr:country tags mix ISO codes and spellings ("GB", "United Kingdom") with the
# REGIONS defaults ("UK"); normalize so one country can't split into two strata.
_COUNTRY_ALIASES = {
    "GB": "UK", "UNITED KINGDOM": "UK",
    "USA": "US", "UNITED STATES": "US", "UNITED STATES OF AMERICA": "US",
    "CANADA": "CA", "AUSTRALIA": "AU",
}


def norm_country(value: str) -> str:
    v = value.strip().upper()
    return _COUNTRY_ALIASES.get(v, v)


def norm_name(name: str) -> str:
    """Lowercase, punctuation-stripped, whitespace-collapsed name key."""
    return re.sub(r"\s+", " ", _PUNCT_RE.sub(" ", name.lower())).strip()


# ---------------------------------------------------------------------------
# Overpass fetch (the OSM source)
# ---------------------------------------------------------------------------
def overpass_query(bbox: tuple) -> str:
    s, w, n, e = bbox
    coords = f"{s},{w},{n},{e}"
    return (
        f"[out:json][timeout:{OVERPASS_TIMEOUT_S}];\n"
        f"(\n"
        f'  node["amenity"="restaurant"]["name"]({coords});\n'
        f'  way["amenity"="restaurant"]["name"]({coords});\n'
        f");\n"
        f"out tags center;\n"
    )


def fetch_region(endpoint: str, region_key: str, cfg: dict) -> list | None:
    """Fetch one region's elements, with exponential backoff on throttling.

    Returns the element list, or None if the region failed all attempts.
    """
    query = overpass_query(cfg["bbox"])
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = requests.post(endpoint, data={"data": query}, timeout=HTTP_TIMEOUT_S)
            if resp.status_code in (429, 504):
                raise requests.RequestException(f"HTTP {resp.status_code}")
            resp.raise_for_status()
            elements = resp.json().get("elements", [])
            print(f"[{region_key}] fetched {len(elements)} elements")
            return elements
        except (requests.RequestException, ValueError) as exc:
            if attempt == MAX_ATTEMPTS:
                print(f"[{region_key}] FAILED after {MAX_ATTEMPTS} attempts ({exc}); skipping region")
                return None
            wait = BACKOFF_BASE_S * (3 ** (attempt - 1))
            print(f"[{region_key}] attempt {attempt}/{MAX_ATTEMPTS} failed ({exc}); retrying in {wait:.0f}s")
            time.sleep(wait)
    return None


def element_to_row(el: dict, cfg: dict) -> dict | None:
    """Build a TRANSIENT harvest row from an Overpass element, or None to skip.

    The row carries lean persisted fields (restaurant_id/name/city/source/is_chain)
    PLUS transient-only fields the harvest itself needs: `lat`/`lng` (dedup +
    Places locationBias) and `cuisine` (stratified-sampling buckets). Only the
    lean fields are written to corpus.sqlite (see main); the rest are dropped.
    """
    tags = el.get("tags", {})
    name = tags.get("name", "").strip()
    if not name:
        return None
    # Skip clearly closed/former places that still carry amenity=restaurant.
    if any(tags.get(k) for k in ("disused", "abandoned", "was:amenity", "disused:amenity")):
        return None
    if el["type"] == "node":
        lat, lng = el.get("lat"), el.get("lon")
    else:  # way -> `out center` supplies a center dict
        center = el.get("center") or {}
        lat, lng = center.get("lat"), center.get("lon")
    if lat is None or lng is None:
        return None
    city = tags.get("addr:city", "").strip() or cfg["city"]
    cuisine = [c.strip().lower() for c in tags.get("cuisine", "").split(";") if c.strip()]
    return {
        # id via the shared corpus helper (name+city) -- NOT harvest's own hashing,
        # so a re-harvest from any source produces stable, matching ids.
        "restaurant_id": restaurant_id_for(name, city),
        "name": name,
        "city": city,
        "source": "osm",
        "is_chain": bool(tags.get("brand") or tags.get("brand:wikidata")),
        # transient-only (not persisted):
        "lat": lat,
        "lng": lng,
        "cuisine": cuisine,
        "price_tier": 0,
    }


def harvest_osm(region_keys: list[str], endpoint: str) -> tuple[list, dict, list]:
    """OSM/Overpass source: fetch every region -> (rows_with_region, counts, failed).

    `rows_with_region` is a list of (transient_row, harvest_region_key) tuples --
    the shape a future source (yelp/foursquare) must also return so the dedup /
    sampling / write pipeline below stays source-agnostic.
    """
    rows_with_region: list = []
    region_fetch_counts: dict = {}
    failed_regions: list = []
    for i, rkey in enumerate(region_keys):
        if i > 0:
            time.sleep(SLEEP_BETWEEN_REGIONS_S)
        elements = fetch_region(endpoint, rkey, REGIONS[rkey])
        if elements is None:
            failed_regions.append(rkey)
            continue
        n_before = len(rows_with_region)
        for el in elements:
            row = element_to_row(el, REGIONS[rkey])
            if row is not None:
                rows_with_region.append((row, rkey))
        region_fetch_counts[rkey] = len(rows_with_region) - n_before
        print(f"[{rkey}] kept {region_fetch_counts[rkey]} named restaurant rows")
    return rows_with_region, region_fetch_counts, failed_regions


# Source seam: name -> harvester. Only OSM today; add a source by implementing the
# same (rows_with_region, region_fetch_counts, failed_regions) return contract.
SOURCES = {"osm": harvest_osm}


# ---------------------------------------------------------------------------
# Dedup + chain flags
# ---------------------------------------------------------------------------
def dedup_and_flag_chains(rows_with_region: list) -> list:
    """Dedup by restaurant_id, then by normalized name+city; finalize is_chain.

    `rows_with_region` is a list of (row, harvest_region_key) tuples; returns
    the same shape, deduped, with the name-frequency chain heuristic applied.
    """
    # 1. Dedup by restaurant_id (same name+city across regions/reruns).
    by_id = {}
    for row, rkey in rows_with_region:
        by_id.setdefault(row["restaurant_id"], (row, rkey))
    id_deduped = list(by_id.values())

    # 2. Chain heuristic across the WHOLE harvest: same normalized name at
    #    >= CHAIN_MIN_LOCATIONS distinct locations (distinct restaurant_ids).
    locations_per_name = defaultdict(set)
    for row, _ in id_deduped:
        locations_per_name[norm_name(row["name"])].add(row["restaurant_id"])
    for row, _ in id_deduped:
        if len(locations_per_name[norm_name(row["name"])]) >= CHAIN_MIN_LOCATIONS:
            row["is_chain"] = True

    # 3. Drop rows whose normalized name+city duplicates an already-kept row
    #    (the same restaurant mapped as both a node and a way).
    kept, seen = [], set()
    for row, rkey in id_deduped:
        key = (norm_name(row["name"]), row["city"].lower())
        if key in seen:
            continue
        seen.add(key)
        kept.append((row, rkey))
    return kept


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------
def allocate(counts: dict, total: int) -> dict:
    """Largest-remainder proportional allocation of `total` across strata.

    Each stratum's allocation never exceeds its pool size; allocations sum to
    min(total, sum(counts)). Deterministic (ties broken by stratum key).
    """
    pool = sum(counts.values())
    total = min(total, pool)
    if total == 0:
        return {k: 0 for k in counts}
    quotas = {k: total * c / pool for k, c in counts.items()}
    alloc = {k: int(q) for k, q in quotas.items()}
    remainder = total - sum(alloc.values())
    order = sorted(
        counts,
        key=lambda k: (quotas[k] - alloc[k], counts[k], str(k)),
        reverse=True,
    )
    i = 0
    while remainder > 0 and i < 10 * len(order):
        k = order[i % len(order)]
        if alloc[k] < counts[k]:
            alloc[k] += 1
            remainder -= 1
        i += 1
    return alloc


def cuisine_bucket(row: dict, top_cuisines: set) -> str:
    if not row["cuisine"]:
        return "unspecified"
    primary = row["cuisine"][0]
    return primary if primary in top_cuisines else "other"


def stratified_sample(rows_with_region: list, target: int, top_cuisines: set, rng) -> list:
    """Sample down to ~target across (chain, region, cuisine), >=70% independents."""
    indep = [(r, k) for r, k in rows_with_region if not r["is_chain"]]
    chain = [(r, k) for r, k in rows_with_region if r["is_chain"]]
    n_total = min(target, len(rows_with_region))
    n_chain = min(len(chain), int(round(n_total * (1 - MIN_INDEP_FRAC))))
    n_indep = min(len(indep), n_total - n_chain)
    n_chain = min(len(chain), n_total - n_indep)  # backfill if indep pool is short

    sampled = []
    for pool, n in ((indep, n_indep), (chain, n_chain)):
        strata = defaultdict(list)
        for row, rkey in pool:
            strata[(rkey, cuisine_bucket(row, top_cuisines))].append((row, rkey))
        counts = {k: len(v) for k, v in strata.items()}
        alloc = allocate(counts, n)
        for key in sorted(strata):
            members = sorted(strata[key], key=lambda t: t[0]["restaurant_id"])
            sampled.extend(rng.sample(members, alloc[key]))
    return sampled


# ---------------------------------------------------------------------------
# Optional Google Places enrichment (transient price_tier; is_chain stays OSM)
# ---------------------------------------------------------------------------
_PRICE_LEVEL_MAP = {
    "PRICE_LEVEL_FREE": 0,
    "PRICE_LEVEL_INEXPENSIVE": 1,
    "PRICE_LEVEL_MODERATE": 2,
    "PRICE_LEVEL_EXPENSIVE": 3,
    "PRICE_LEVEL_VERY_EXPENSIVE": 4,
}


def enrich_places(rows: list, api_key: str) -> None:
    """Fill the TRANSIENT price_tier via Places (New) Text Search.

    price_tier is no longer a persisted column in v2's lean schema, so this only
    refines the sampling/report signal (a seam kept from v1 -- see plan §9.1,
    "is_chain may come from OSM tags or Places"). is_chain currently stays sourced
    from OSM brand tags + the name-frequency heuristic; a Places-derived chain
    signal would slot in here.
    """
    url = "https://places.googleapis.com/v1/places:searchText"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "places.priceLevel",
    }
    enriched = 0
    for row in rows:
        body = {
            "textQuery": f"{row['name']}, {row['city']}",
            "pageSize": 1,
            "locationBias": {
                "circle": {
                    "center": {"latitude": row["lat"], "longitude": row["lng"]},
                    "radius": 500.0,
                }
            },
        }
        try:
            resp = requests.post(url, json=body, headers=headers, timeout=30)
            resp.raise_for_status()
            places = resp.json().get("places", [])
            level = places[0].get("priceLevel") if places else None
            if level in _PRICE_LEVEL_MAP:
                row["price_tier"] = _PRICE_LEVEL_MAP[level]
                enriched += 1
        except (requests.RequestException, ValueError, KeyError) as exc:
            print(f"  places enrichment failed for {row['name']!r}: {exc}")
        time.sleep(0.1)  # stay well under Places QPS limits
    print(f"Places enrichment: price_tier set on {enriched}/{len(rows)} rows")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_report(sampled: list, region_fetch_counts: dict, failed_regions: list) -> None:
    print("\n===== distribution report =====")
    print(f"total sampled rows: {len(sampled)}")

    region_counts = Counter(rkey for _, rkey in sampled)
    print("\nper-region (sampled / fetched):")
    for rkey in sorted(region_fetch_counts):
        print(f"  {rkey:<14} {region_counts.get(rkey, 0):>5} / {region_fetch_counts[rkey]}")
    for rkey in failed_regions:
        print(f"  {rkey:<14} SKIPPED (fetch failed)")

    cuisines = Counter()
    for row, _ in sampled:
        if row["cuisine"]:
            cuisines[row["cuisine"][0]] += 1
        else:
            cuisines["(unspecified)"] += 1
    print("\ntop cuisines (primary tag, sampled rows):")
    for name, count in cuisines.most_common(15):
        print(f"  {name:<20} {count:>5}  ({100 * count / len(sampled):.1f}%)")

    n_chain = sum(1 for row, _ in sampled if row["is_chain"])
    n_indep = len(sampled) - n_chain
    print(
        f"\nchain vs independent: {n_chain} chain / {n_indep} independent "
        f"({100 * n_indep / len(sampled):.1f}% independent)"
    )
    print("=" * 31)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_fractions(text: str) -> dict[str, float]:
    """Parse "sft=0.5,grpo=0.3,eval=0.2" -> {split: fraction} (validated by assign_splits)."""
    fractions: dict[str, float] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise argparse.ArgumentTypeError(f"bad --fractions entry {part!r} (want split=fraction)")
        split, value = part.split("=", 1)
        split = split.strip()
        if split not in VALID_SPLITS:
            raise argparse.ArgumentTypeError(f"unknown split {split!r} in --fractions")
        try:
            fractions[split] = float(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"non-numeric fraction {value!r} for {split!r}")
    return fractions or dict(DEFAULT_FRACTIONS)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=sorted(SOURCES), default="osm",
                        help="harvest source (default osm; the only one implemented)")
    parser.add_argument(
        "--regions", nargs="+", metavar="NAME", default=None,
        help=f"subset of regions to harvest (default: all). Available: {', '.join(REGIONS)}",
    )
    parser.add_argument("--target", type=int, default=3500, help="stratified-sample size (default 3500)")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="Overpass API endpoint (point at a mirror if throttled)")
    parser.add_argument("--enrich-places", action="store_true", help="enrich (transient) price_tier via Google Places (needs GOOGLE_PLACES_API_KEY)")
    parser.add_argument("--db", type=Path, default=REPO_ROOT / "data" / "corpus.sqlite",
                        help="corpus.sqlite path (default data/corpus.sqlite)")
    parser.add_argument("--assign-split", choices=["none", "random"], default="none",
                        help="none (default): leave rows unmarked (use assign_splits.py later); "
                             "random: fill the split of currently-unmarked rows in this run (seeded)")
    parser.add_argument("--fractions", type=parse_fractions, default=None,
                        help='with --assign-split random, per-split shares as '
                             '"sft=0.5,grpo=0.3,eval=0.2" (default 50/30/20)')
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for sampling + split (default 42)")
    return parser.parse_args()


def main():
    import random

    args = parse_args()
    if args.regions:
        unknown = [r for r in args.regions if r not in REGIONS]
        if unknown:
            sys.exit(f"unknown region(s): {', '.join(unknown)} (available: {', '.join(REGIONS)})")
        region_keys = args.regions
    else:
        region_keys = list(REGIONS)

    # 1. Harvest via the selected source seam.
    harvester = SOURCES[args.source]
    rows_with_region, region_fetch_counts, failed_regions = harvester(region_keys, args.endpoint)
    if not rows_with_region:
        sys.exit("no rows harvested from any region; aborting")
    print(f"\nharvested {len(rows_with_region)} rows from {len(region_fetch_counts)} region(s)"
          + (f" ({len(failed_regions)} failed: {', '.join(failed_regions)})" if failed_regions else ""))

    # 2. Dedup + chain flags.
    deduped = dedup_and_flag_chains(rows_with_region)
    print(f"after dedup (id, then name+city): {len(deduped)} rows")

    # 3. Cuisine buckets from the full deduped pool (long tail -> 'other').
    cuisine_counts = Counter(r["cuisine"][0] for r, _ in deduped if r["cuisine"])
    top_cuisines = {name for name, _ in cuisine_counts.most_common(TOP_CUISINES)}

    # 4. Stratified sample (geography x cuisine x chain, >=70% independents).
    rng = random.Random(args.seed)
    sampled = stratified_sample(deduped, args.target, top_cuisines, rng)
    print(f"after stratified sampling: {len(sampled)} rows (target {args.target})")

    # 5. Optional Places enrichment on the sampled rows only (bounds API usage).
    if args.enrich_places:
        api_key = os.environ.get("GOOGLE_PLACES_API_KEY", "").strip()
        if api_key:
            enrich_places([row for row, _ in sampled], api_key)
        else:
            print("GOOGLE_PLACES_API_KEY not set; skipping Places enrichment")

    # 6. Write LEAN rows to corpus.sqlite. Omit 'split' so a re-harvest never wipes
    #    an existing assignment (upsert_restaurants preserves it).
    with open_corpus(args.db) as cx:
        lean = [
            {"name": r["name"], "city": r["city"], "source": r["source"], "is_chain": r["is_chain"]}
            for r, _ in sampled
        ]
        n = cx.upsert_restaurants(lean)
        cx.set_meta("harvest_source", args.source)
        print(f"\nupserted {n} restaurants into {args.db}")

        # 7. Optional in-run split marking (fills only unmarked rows -- for a full
        #    re-shuffle use assign_splits.py --reassign).
        if args.assign_split == "random":
            fractions = args.fractions or dict(DEFAULT_FRACTIONS)
            assigned = cx.assign_splits(seed=args.seed, fractions=fractions)
            print(f"assigned split (seed {args.seed}, fractions {fractions}): "
                  + ", ".join(f"{s}={assigned.get(s, 0)}" for s in VALID_SPLITS if s in fractions))
        counts = cx.count_by_split()
        print("split counts: " + "  ".join(f"{s}={counts.get(s, 0)}"
                                            for s in (*VALID_SPLITS, "unmarked")))

    # 8. Distribution report.
    print_report(sampled, region_fetch_counts, failed_regions)


if __name__ == "__main__":
    main()
