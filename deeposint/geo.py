"""Small geodesy helpers (WGS84, good enough for distances < ~50 km)."""

from __future__ import annotations

import math

M_PER_DEG_LAT = 111_320.0


def dist_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Equirectangular distance in metres between (lat, lon) pairs."""
    dy = (a[0] - b[0]) * M_PER_DEG_LAT
    dx = (a[1] - b[1]) * M_PER_DEG_LAT * math.cos(math.radians((a[0] + b[0]) / 2))
    return math.hypot(dx, dy)


def bearing_deg(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Initial great-circle bearing from a to b, degrees clockwise from north."""
    la1, la2 = math.radians(a[0]), math.radians(b[0])
    dlon = math.radians(b[1] - a[1])
    y = math.sin(dlon) * math.cos(la2)
    x = math.cos(la1) * math.sin(la2) - math.sin(la1) * math.cos(la2) * math.cos(dlon)
    return math.degrees(math.atan2(y, x)) % 360


def tile_xy(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    """Slippy-map tile containing (lat, lon)."""
    n = 2**zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_r = math.radians(lat)
    y = int((1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n)
    return x, y


def angle_diff(a: float, b: float) -> float:
    """Smallest absolute difference between two headings, degrees."""
    return abs((a - b + 180.0) % 360.0 - 180.0)
