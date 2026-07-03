"""WS-B: harvest a stratified restaurant corpus from OSM/Overpass.

Queries the Overpass API for named `amenity=restaurant` elements (nodes + ways
with center coords) across a hard-coded list of English-speaking metro bounding
boxes (US/CA/UK/AU), dedups them, optionally enriches with Google Places
(price tier), stratified-samples down to a target size (oversampling
independents), and writes:

  data/restaurants.jsonl   one contract-1.4 row per line (see notes/phase2_plan.md)
  data/splits.json         {restaurant_id: "train" | "eval"}, stratified + disjoint

Usage:
  uv run python scripts/harvest_restaurants.py                       # full default harvest
  uv run python scripts/harvest_restaurants.py --regions seattle --target 50
  uv run python scripts/harvest_restaurants.py --endpoint https://overpass.kumi.systems/api/interpreter
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import requests
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent

# Load GOOGLE_PLACES_API_KEY (for --enrich-places) from the repo-root .env,
# same convention as src/gemma/run_agent.py / src/claude/run_claude.py.
load_dotenv(REPO_ROOT / ".env")

DEFAULT_ENDPOINT = "https://overpass-api.de/api/interpreter"
OVERPASS_TIMEOUT_S = 180          # server-side [timeout:...] in the query
HTTP_TIMEOUT_S = OVERPASS_TIMEOUT_S + 30
SLEEP_BETWEEN_REGIONS_S = 3.0     # politeness gap between region queries
MAX_ATTEMPTS = 4                  # per region: 1 try + 3 retries
BACKOFF_BASE_S = 10.0             # 10s, 30s, 90s exponential backoff
TOP_CUISINES = 15                 # cuisines kept as their own stratum; rest -> "other"
MIN_INDEP_FRAC = 0.70             # sampling aims for >= this fraction independents
CHAIN_MIN_LOCATIONS = 3           # name-frequency heuristic threshold for is_chain

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

# addr:country tags mix ISO codes and spellings ("GB", "United Kingdom") with
# the REGIONS defaults ("UK"); normalize so one country can't split into two
# strata (observed: 799 "UK" + 80 "GB" rows in the same harvest).
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


def restaurant_id(name: str, lat: float, lng: float) -> str:
    """Contractual id (1.4): 5-decimal lat/lng formatting is load-bearing."""
    return hashlib.sha1(f"{name}|{lat:.5f}|{lng:.5f}".encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Overpass fetch
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


# ---------------------------------------------------------------------------
# Row construction (contract 1.4)
# ---------------------------------------------------------------------------

def element_to_row(el: dict, cfg: dict) -> dict | None:
    """Build a contract-1.4 row from an Overpass element, or None to skip."""
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
    cuisine = [c.strip().lower() for c in tags.get("cuisine", "").split(";") if c.strip()]
    return {
        "restaurant_id": restaurant_id(name, lat, lng),
        "name": name,
        "city": tags.get("addr:city", "").strip() or cfg["city"],
        "region": (tags.get("addr:state", "") or tags.get("addr:province", "")).strip() or cfg["region"],
        "country": norm_country(tags.get("addr:country", "")) or cfg["country"],
        "lat": lat,
        "lng": lng,
        "cuisine": cuisine,
        "price_tier": 0,
        "is_chain": bool(tags.get("brand") or tags.get("brand:wikidata")),
        "source": "osm",
    }


def dedup_and_flag_chains(rows_with_region: list) -> list:
    """Dedup by restaurant_id, then by normalized name+city; finalize is_chain.

    `rows_with_region` is a list of (row, harvest_region_key) tuples; returns
    the same shape, deduped, with the name-frequency chain heuristic applied.
    """
    # 1. Dedup by restaurant_id (identical name+coords across regions/reruns).
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
# Stratified sampling & split
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


def stratified_split(sampled: list, eval_frac: float, top_cuisines: set, rng) -> dict:
    """Disjoint train/eval split, stratified over (region, cuisine, chain)."""
    strata = defaultdict(list)
    for row, rkey in sampled:
        strata[(rkey, cuisine_bucket(row, top_cuisines), row["is_chain"])].append(row)
    counts = {k: len(v) for k, v in strata.items()}
    eval_alloc = allocate(counts, int(round(eval_frac * len(sampled))))
    splits = {}
    for key in sorted(strata, key=str):
        members = sorted(strata[key], key=lambda r: r["restaurant_id"])
        rng.shuffle(members)
        n_eval = eval_alloc[key]
        for i, row in enumerate(members):
            splits[row["restaurant_id"]] = "eval" if i < n_eval else "train"
    return splits


# ---------------------------------------------------------------------------
# Optional Google Places enrichment
# ---------------------------------------------------------------------------

_PRICE_LEVEL_MAP = {
    "PRICE_LEVEL_FREE": 0,
    "PRICE_LEVEL_INEXPENSIVE": 1,
    "PRICE_LEVEL_MODERATE": 2,
    "PRICE_LEVEL_EXPENSIVE": 3,
    "PRICE_LEVEL_VERY_EXPENSIVE": 4,
}


def enrich_places(rows: list, api_key: str) -> None:
    """Fill price_tier via Places (New) Text Search; mark enriched rows 'places'."""
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
                row["source"] = "places"
                enriched += 1
        except (requests.RequestException, ValueError, KeyError) as exc:
            print(f"  places enrichment failed for {row['name']!r}: {exc}")
        time.sleep(0.1)  # stay well under Places QPS limits
    print(f"Places enrichment: price_tier set on {enriched}/{len(rows)} rows")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(sampled: list, splits: dict, region_fetch_counts: dict, failed_regions: list) -> None:
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

    split_counts = Counter(splits.values())
    print(f"\nsplit sizes: train={split_counts.get('train', 0)}  eval={split_counts.get('eval', 0)}")
    print("=" * 31)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--regions", nargs="+", metavar="NAME", default=None,
        help=f"subset of regions to harvest (default: all). Available: {', '.join(REGIONS)}",
    )
    parser.add_argument("--target", type=int, default=3500, help="stratified-sample size (default 3500)")
    parser.add_argument("--eval-frac", type=float, default=0.143, help="eval fraction of the sample (default 0.143)")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="Overpass API endpoint (point at a mirror if throttled)")
    parser.add_argument("--enrich-places", action="store_true", help="enrich price_tier via Google Places (needs GOOGLE_PLACES_API_KEY)")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "data"), help="output directory (default data/)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for sampling/splits (default 42)")
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

    # 1. Harvest.
    rows_with_region = []
    region_fetch_counts = {}
    failed_regions = []
    for i, rkey in enumerate(region_keys):
        if i > 0:
            time.sleep(SLEEP_BETWEEN_REGIONS_S)
        elements = fetch_region(args.endpoint, rkey, REGIONS[rkey])
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

    # 6. Stratified disjoint train/eval split.
    splits = stratified_split(sampled, args.eval_frac, top_cuisines, rng)

    # 7. Write outputs (rows sorted by restaurant_id for stable diffs).
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "restaurants.jsonl"
    splits_path = out_dir / "splits.json"
    with open(rows_path, "w", encoding="utf-8") as f:
        for row, _ in sorted(sampled, key=lambda t: t[0]["restaurant_id"]):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(splits_path, "w", encoding="utf-8") as f:
        json.dump(dict(sorted(splits.items())), f, indent=2)
    print(f"\nwrote {len(sampled)} rows to {rows_path}")
    print(f"wrote {len(splits)} split entries to {splits_path}")

    # 8. Distribution report.
    print_report(sampled, splits, region_fetch_counts, failed_regions)


if __name__ == "__main__":
    main()
