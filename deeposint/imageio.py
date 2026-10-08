"""Image loading (with optional HEIC) and EXIF GPS extraction."""

from __future__ import annotations

from pathlib import Path

from PIL import ExifTags, Image

try:  # optional HEIC support
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:  # pragma: no cover
    pass


def open_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _to_deg(v) -> float:
    d, m, s = (float(x) for x in v)
    return d + m / 60 + s / 3600


def exif_gps(path: str | Path) -> tuple[float, float] | None:
    """(lat, lon) from EXIF if present — the cheapest possible answer."""
    try:
        exif = Image.open(path).getexif()
        gps = exif.get_ifd(ExifTags.IFD.GPSInfo)
    except Exception:
        return None
    if not gps or 2 not in gps or 4 not in gps:
        return None
    lat, lon = _to_deg(gps[2]), _to_deg(gps[4])
    if gps.get(1) == "S":
        lat = -lat
    if gps.get(3) == "W":
        lon = -lon
    return lat, lon
