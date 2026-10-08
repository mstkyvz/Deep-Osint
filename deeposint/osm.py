"""OpenStreetMap access: area name -> bbox (Nominatim), bbox -> road geometries (Overpass)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode

from . import http
from .geo import bearing_deg, dist_m

OVERPASS_ENDPOINTS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]

# Road classes worth scanning for street-level photos. Residential streets are
# opt-in because they multiply the gallery size ~5-10x.
DEFAULT_HIGHWAYS = ("trunk", "primary", "secondary", "tertiary", "trunk_link", "primary_link")


@dataclass
class Road:
    osm_id: int
    name: str
    highway: str
    coords: list[tuple[float, float]]


@dataclass
class Sample:
    lat: float
    lon: float
    road_bearing: float
    road: str


def area_bbox(name: str, cache_dir: Path | None = None) -> tuple[float, float, float, float]:
    """Resolve 'Karamürsel, Kocaeli' style names to (south, west, north, east)."""
    q = urlencode({"q": name, "format": "json", "limit": 1, "polygon_geojson": 0})
    raw = http.get(f"https://nominatim.openstreetmap.org/search?{q}", cache_dir=cache_dir)
    res = json.loads(raw)
    if not res:
        raise ValueError(f"Nominatim found nothing for {name!r}")
    s, n, w, e = map(float, res[0]["boundingbox"])
    return s, w, n, e


def fetch_roads(bbox: tuple[float, float, float, float], highways: tuple[str, ...] = DEFAULT_HIGHWAYS,
                cache_dir: Path | None = None) -> list[Road]:
    s, w, n, e = bbox
    hw = "|".join(highways)
    ql = f'[out:json][timeout:120];way["highway"~"^({hw})$"]({s},{w},{n},{e});out geom;'
    last: Exception | None = None
    for ep in OVERPASS_ENDPOINTS:
        try:
            raw = http.get(ep, method="POST", data={"data": ql}, cache_dir=cache_dir, timeout=180, retries=2)
            elements = json.loads(raw)["elements"]
            break
        except (RuntimeError, ValueError, KeyError) as e:
            last = e
    else:
        raise RuntimeError("all Overpass endpoints failed") from last
    roads = []
    for el in elements:
        geom = el.get("geometry") or []
        if len(geom) < 2:
            continue
        tags = el.get("tags", {})
        roads.append(Road(el["id"], tags.get("name", ""), tags.get("highway", ""),
                          [(p["lat"], p["lon"]) for p in geom]))
    return roads


def sample_roads(roads: list[Road], step_m: float) -> list[Sample]:
    """Points every step_m metres along each road, carrying the local road bearing."""
    out: list[Sample] = []
    for r in roads:
        carry = 0.0  # distance already travelled since the last sample
        for a, b in zip(r.coords, r.coords[1:]):
            seg = dist_m(a, b)
            if seg == 0:
                continue
            brg = bearing_deg(a, b)
            t = (step_m - carry) % step_m if carry else 0.0
            while t <= seg:
                f = t / seg
                out.append(Sample(a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, brg, r.name))
                t += step_m
            carry = (carry + seg) % step_m
    return out
