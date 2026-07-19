"""Unit tests for the Places source transforms in scripts/corpus/harvest.py.

Network-free: only the pure place->row logic + query construction are exercised.

Run: uv run python -m pytest tests/test_harvest.py -q
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "corpus"))

import harvest as h  # noqa: E402

CFG = {"city": "Seattle", "region": "Washington"}


def _place(name="Joe's", status="OPERATIONAL", ratings=50, ptype="italian_restaurant",
           locality="Seattle", website="http://x"):
    return {
        "displayName": {"text": name},
        "businessStatus": status,
        "userRatingCount": ratings,
        "primaryType": ptype,
        "addressComponents": [{"types": ["locality"], "longText": locality}] if locality else [],
        "location": {"latitude": 47.6, "longitude": -122.3},
        "websiteUri": website,
    }


def test_place_to_row_basic():
    r = h.place_to_row(_place(), CFG)
    assert r["name"] == "Joe's" and r["city"] == "Seattle" and r["source"] == "places"
    assert r["is_chain"] is False and r["cuisine"] == ["italian"]
    assert r["website"] == "http://x"
    assert r["restaurant_id"] == h.restaurant_id_for("Joe's", "Seattle")


def test_place_to_row_skips_unnamed():
    assert h.place_to_row(_place(name="  "), CFG) is None


def test_place_to_row_drops_closed():
    assert h.place_to_row(_place(status="CLOSED_PERMANENTLY"), CFG) is None
    assert h.place_to_row(_place(status="CLOSED_TEMPORARILY"), CFG) is None
    # a missing businessStatus is allowed (treated as operational)
    assert h.place_to_row(_place(status=None), CFG) is not None


def test_place_to_row_min_ratings():
    assert h.place_to_row(_place(ratings=50), CFG, min_ratings=3) is not None
    assert h.place_to_row(_place(ratings=2), CFG, min_ratings=3) is None
    assert h.place_to_row(_place(ratings=0), CFG, min_ratings=3) is None
    # missing rating count counts as 0
    p = _place(); del p["userRatingCount"]
    assert h.place_to_row(p, CFG, min_ratings=3) is None
    # min_ratings=0 (default) disables the floor
    assert h.place_to_row(_place(ratings=0), CFG, min_ratings=0) is not None


def test_place_to_row_city_fallback_to_cfg():
    assert h.place_to_row(_place(locality=None), CFG)["city"] == "Seattle"


def test_cuisine_from_primary_type():
    assert h.place_to_row(_place(ptype="seafood_restaurant"), CFG)["cuisine"] == ["seafood"]
    assert h.place_to_row(_place(ptype="cafe"), CFG)["cuisine"] == ["cafe"]
    assert h.place_to_row(_place(ptype="restaurant"), CFG)["cuisine"] == []
    assert h.place_to_row(_place(ptype=""), CFG)["cuisine"] == []


def test_locality_extraction():
    assert h._locality({"addressComponents": [
        {"types": ["route"], "longText": "Main St"},
        {"types": ["locality", "political"], "longText": "Bellevue"},
    ]}) == "Bellevue"
    assert h._locality({"addressComponents": []}) is None


def test_field_mask_and_type_filter_present():
    # regression guards for the data-quality wiring
    assert "userRatingCount" in h.PLACES_FIELD_MASK
    assert "businessStatus" in h.PLACES_FIELD_MASK
    assert "websiteUri" in h.PLACES_FIELD_MASK


def test_places_registered_as_source():
    assert "places" in h.SOURCES and h.SOURCES["places"] is h.harvest_places
