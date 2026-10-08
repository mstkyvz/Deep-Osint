"""Street View gallery builder.

Google retired the `GeoPhotoService.SingleImageSearch` endpoint that most
open-source tools used for "nearest panorama" lookups. This module instead:

1. lists panoramas per z17 map tile via the coverage endpoint
   (`/maps/photometa/ac/v1`) used by the Maps web client, then
2. renders perspective crops with the `geo0.ggpht.com/cbk?output=thumbnail` endpoint.

Both are undocumented and may change or be rate-limited; respect Google's terms
for your use case. For a supported path use the Street View Static API (key,
paid) or Mapillary (free, sparser coverage in Turkey) — see the provider TODO in
the README.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from tqdm import tqdm

from . import http
from .geo import dist_m, tile_xy
from .osm import Sample

COVERAGE_URL = "https://www.google.com/maps/photometa/ac/v1?pb=!1m1!1smaps_sv.tactile!6m3!1i{x}!2i{y}!3i17!8b1"
THUMB_URL = ("https://geo0.ggpht.com/cbk?cb_client=maps_sv.tactile&output=thumbnail&panoid={pid}"
             "&w={w}&h={h}&yaw={yaw:.1f}&pitch={pitch:.1f}&thumbfov={fov:.0f}")


@dataclass
class Pano:
    pano_id: str
    lat: float
    lon: float


@dataclass
class View:
    view_id: str
    pano_id: str
    lat: float
    lon: float
    heading: float
    road: str
    path: str


def parse_coverage(raw: bytes) -> list[Pano]:
    """Parse a coverage tile response: ")]}'\\n" + JSON, panos under d[1][1]."""
    text = raw.decode("utf-8", "replace")
    try:
        d = json.loads(text[text.index("\n") + 1:])
        entries = d[1][1] or []
    except (ValueError, IndexError, TypeError):
        return []
    out = []
    for p in entries:
        try:
            pid = p[0][0][1]
            ll = p[0][2][0]
            out.append(Pano(pid, float(ll[2]), float(ll[3])))
        except (IndexError, TypeError, ValueError):
            continue
    return out


def panos_for_samples(samples: list[Sample], cache_dir: Path, workers: int = 12) -> dict[str, Pano]:
    tiles = sorted({tile_xy(s.lat, s.lon, 17) for s in samples})

    def fetch(t: tuple[int, int]) -> list[Pano]:
        try:
            return parse_coverage(http.get(COVERAGE_URL.format(x=t[0], y=t[1]), cache_dir=cache_dir))
        except RuntimeError:
            return []

    panos: dict[str, Pano] = {}
    with ThreadPoolExecutor(workers) as ex:
        for lst in tqdm(ex.map(fetch, tiles), total=len(tiles), desc="coverage tiles"):
            for p in lst:
                panos[p.pano_id] = p
    return panos


def assign_panos(samples: list[Sample], panos: dict[str, Pano], max_dist_m: float = 25.0
                 ) -> list[tuple[Pano, Sample]]:
    """Nearest pano per sample (within max_dist_m), de-duplicated."""
    grid: dict[tuple[float, float], list[Pano]] = {}
    for p in panos.values():
        grid.setdefault((round(p.lat, 3), round(p.lon, 3)), []).append(p)
    chosen: dict[str, tuple[Pano, Sample]] = {}
    for s in samples:
        best: tuple[float, Pano] | None = None
        for dla in (-0.001, 0.0, 0.001):
            for dlo in (-0.001, 0.0, 0.001):
                for p in grid.get((round(s.lat + dla, 3), round(s.lon + dlo, 3)), []):
                    d = dist_m((s.lat, s.lon), (p.lat, p.lon))
                    if d <= max_dist_m and (best is None or d < best[0]):
                        best = (d, p)
        if best and best[1].pano_id not in chosen:
            chosen[best[1].pano_id] = (best[1], s)
    return list(chosen.values())


def render_views(pairs: list[tuple[Pano, Sample]], out_dir: Path, offsets: tuple[float, ...] = (-35, 0, 35),
                 sides: tuple[float, ...] = (90, 270), size: tuple[int, int] = (640, 480),
                 fov: float = 80, pitch: float = 8, workers: int = 16) -> list[View]:
    """Render views perpendicular to the road (both sides) with small yaw offsets."""
    img_dir = out_dir / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for pano, s in pairs:
        for side in sides:
            for off in offsets:
                h = (s.road_bearing + side + off) % 360
                vid = f"{pano.pano_id}_{h:05.1f}".replace("-", "~")
                jobs.append(View(vid, pano.pano_id, pano.lat, pano.lon, round(h, 1), s.road,
                                 str(img_dir / f"{vid}.jpg")))

    def render(v: View) -> View | None:
        p = Path(v.path)
        if p.exists() and p.stat().st_size > 0:
            return v
        url = THUMB_URL.format(pid=v.pano_id, w=size[0], h=size[1], yaw=v.heading, pitch=pitch, fov=fov)
        try:
            body = http.get(url)
        except RuntimeError:
            return None
        if body[:2] != b"\xff\xd8":  # not a JPEG (error JSON, blocked, ...)
            return None
        p.write_bytes(body)
        return v

    views: list[View] = []
    with ThreadPoolExecutor(workers) as ex:
        for v in tqdm(ex.map(render, jobs), total=len(jobs), desc="render views"):
            if v is not None:
                views.append(v)
    (out_dir / "views.json").write_text(json.dumps([asdict(v) for v in views], ensure_ascii=False))
    return views


def load_views(out_dir: Path) -> list[View]:
    return [View(**d) for d in json.loads((out_dir / "views.json").read_text())]
