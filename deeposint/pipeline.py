"""End-to-end street-level pipeline: area + photo -> ranked locations."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from . import osm, streetview
from .imageio import exif_gps
from .rerank import Candidate, Reranker, cluster_candidates, confidence, save_ranking

log = logging.getLogger("deeposint")


@dataclass
class Config:
    work_dir: Path
    step_m: float = 40.0
    highways: tuple[str, ...] = osm.DEFAULT_HIGHWAYS
    offsets: tuple[float, ...] = (-35.0, 0.0, 35.0)
    retrieval_backend: str = "megaloc"
    verify_backend: str = "lightglue"
    device: str | None = None
    recall_k: int = 200
    cluster_radius_m: float = 40.0
    min_inliers: int = 15
    reranker_model: Path | None = None
    query_box: tuple[int, int, int, int] | None = None


def build_gallery(bbox: tuple[float, float, float, float], cfg: Config) -> list[streetview.View]:
    """OSM roads -> samples -> panos -> rendered views. Cached under cfg.work_dir."""
    gdir = cfg.work_dir / "gallery"
    if (gdir / "views.json").exists():
        log.info("reusing gallery at %s", gdir)
        return streetview.load_views(gdir)
    cache = cfg.work_dir / "http_cache"
    roads = osm.fetch_roads(bbox, cfg.highways, cache_dir=cache)
    samples = osm.sample_roads(roads, cfg.step_m)
    log.info("%d roads, %d samples", len(roads), len(samples))
    panos = streetview.panos_for_samples(samples, cache_dir=cache)
    pairs = streetview.assign_panos(samples, panos)
    log.info("%d panos in tiles, %d chosen", len(panos), len(pairs))
    return streetview.render_views(pairs, gdir, offsets=cfg.offsets)


def locate(photo: str, bbox: tuple[float, float, float, float], cfg: Config) -> dict:
    cfg.work_dir.mkdir(parents=True, exist_ok=True)

    gps = exif_gps(photo)
    if gps:
        return {"method": "exif", "lat": gps[0], "lon": gps[1], "confidence": {"level": "high"}}

    views = build_gallery(bbox, cfg)
    if not views:
        raise RuntimeError("empty gallery: no Street View coverage or endpoints blocked")

    # Recall: global descriptors.
    from .retrieval import Embedder, build_index, top_k

    emb = Embedder(cfg.retrieval_backend, cfg.device)
    feats = build_index(emb, [v.path for v in views], cfg.work_dir / f"index_{cfg.retrieval_backend}.npz")
    q = emb.embed_query(photo, cfg.query_box)
    shortlist = top_k(q, feats, cfg.recall_k)

    # Precision: local verification on the shortlist.
    from tqdm import tqdm

    from .verify import Verifier

    ver = Verifier(cfg.verify_backend, cfg.device)
    qf = ver.query_features(photo, cfg.query_box)
    cands: list[Candidate] = []
    for rank, (i, sim) in enumerate(tqdm(shortlist, desc=f"verify ({cfg.verify_backend})")):
        v = views[i]
        res = ver.verify(qf, v.path)
        cands.append(Candidate(v.view_id, v.pano_id, v.lat, v.lon, v.heading, v.road, v.path,
                               global_sim=sim, global_rank=rank, matches=res.matches, inliers=res.inliers))

    # Decision: neighbour voting + re-ranking.
    clusters = cluster_candidates(cands, cfg.cluster_radius_m, cfg.min_inliers)
    clusters = Reranker(cfg.reranker_model).score_clusters(clusters)
    save_ranking(clusters, cfg.work_dir / "ranking.json")

    from .report import contact_sheet

    contact_sheet(photo, clusters, cfg.work_dir / "top.jpg")

    top = clusters[0]
    result = {
        "method": "streetview",
        "lat": top.lat,
        "lon": top.lon,
        "heading": round(top.heading, 1),
        "road": top.best.road,
        "confidence": confidence(clusters),
        "gallery_views": len(views),
        "best_view": top.best.path,
        "ranking": str(cfg.work_dir / "ranking.json"),
        "sheet": str(cfg.work_dir / "top.jpg"),
    }
    (cfg.work_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result
