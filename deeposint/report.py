"""Contact sheet of the query next to the top-ranked candidates."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from .imageio import open_rgb
from .rerank import Cluster


def contact_sheet(query: str, clusters: list[Cluster], out: Path, top: int = 8, tile_w: int = 400) -> None:
    tile_h = int(tile_w * 0.75)
    cols = 3
    n = min(top, len(clusters)) + 1
    rows = (n + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile_w, rows * (tile_h + 22)), "white")
    draw = ImageDraw.Draw(sheet)
    items = [("QUERY", query)] + [
        (f"#{i} s={c.score:.2f} inl={c.best.inliers} sup={c.support} {c.lat:.5f},{c.lon:.5f} h={c.heading:.0f}",
         c.best.path)
        for i, c in enumerate(clusters[:top], 1)
    ]
    for k, (label, path) in enumerate(items):
        x, y = (k % cols) * tile_w, (k // cols) * (tile_h + 22)
        im = open_rgb(path)
        im.thumbnail((tile_w, tile_h))
        sheet.paste(im, (x, y + 22))
        draw.text((x + 4, y + 4), label, fill="black")
    sheet.save(out, quality=90)
