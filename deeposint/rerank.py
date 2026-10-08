"""Neighbour voting and re-ranking (decision stage).

A true location is usually supported by several adjacent panoramas (the same
facades seen from 10-40 m apart), while a false positive tends to be a single
lucky view. We therefore cluster verified candidates spatially and score the
cluster, not the single image.

Scoring:
- default: a transparent heuristic (see ``heuristic_score``);
- optional: a LightGBM model trained with ``deeposint train-reranker`` on
  self-supervised pairs mined from the gallery itself (no manual labels).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .geo import dist_m

FEATURES = ["global_sim", "global_rank", "inliers", "inlier_ratio",
            "cluster_max_inliers", "cluster_support", "cluster_views", "heading_spread"]


@dataclass
class Candidate:
    view_id: str
    pano_id: str
    lat: float
    lon: float
    heading: float
    road: str
    path: str
    global_sim: float
    global_rank: int
    matches: int = 0
    inliers: int = 0
    feats: dict = field(default_factory=dict)
    score: float = 0.0


@dataclass
class Cluster:
    lat: float
    lon: float
    heading: float
    score: float
    support: int
    best: Candidate
    members: list[Candidate]


def cluster_candidates(cands: list[Candidate], radius_m: float = 40.0, min_inliers: int = 15) -> list[Cluster]:
    """Greedy spatial clustering seeded by the strongest verified candidates."""
    order = sorted(cands, key=lambda c: (c.inliers, c.global_sim), reverse=True)
    used: set[str] = set()
    clusters: list[Cluster] = []
    for seed in order:
        if seed.view_id in used:
            continue
        members = [c for c in order if c.view_id not in used
                   and dist_m((seed.lat, seed.lon), (c.lat, c.lon)) <= radius_m]
        for c in members:
            used.add(c.view_id)
        strong = [c for c in members if c.inliers >= min_inliers]
        support = len({c.pano_id for c in strong})
        hs = [c.heading for c in strong] or [seed.heading]
        # circular mean of headings of supporting views
        hx = sum(math.cos(math.radians(h)) for h in hs)
        hy = sum(math.sin(math.radians(h)) for h in hs)
        spread = 1 - math.hypot(hx, hy) / len(hs)
        for c in members:
            c.feats = {
                "global_sim": c.global_sim,
                "global_rank": c.global_rank,
                "inliers": c.inliers,
                "inlier_ratio": c.inliers / c.matches if c.matches else 0.0,
                "cluster_max_inliers": seed.inliers,
                "cluster_support": support,
                "cluster_views": len(members),
                "heading_spread": spread,
            }
        clusters.append(Cluster(seed.lat, seed.lon, math.degrees(math.atan2(hy, hx)) % 360,
                                0.0, support, seed, members))
    return clusters


def heuristic_score(f: dict) -> float:
    """Monotone, easy-to-audit score. Inliers dominate; neighbour support breaks ties."""
    return (math.log1p(f["cluster_max_inliers"]) * 2.0
            + min(f["cluster_support"], 5) * 0.6
            + f["global_sim"] * 1.5
            - min(f["global_rank"], 500) / 500.0 * 0.5)


class Reranker:
    def __init__(self, model_path: Path | None = None):
        self.model = None
        if model_path and Path(model_path).exists():
            import lightgbm as lgb

            self.model = lgb.Booster(model_file=str(model_path))

    def score_clusters(self, clusters: list[Cluster]) -> list[Cluster]:
        for cl in clusters:
            f = cl.best.feats
            if self.model is not None:
                cl.score = float(self.model.predict(np.array([[f[k] for k in FEATURES]]))[0])
            else:
                cl.score = heuristic_score(f)
        return sorted(clusters, key=lambda c: c.score, reverse=True)


def confidence(clusters: list[Cluster]) -> dict:
    """Margin-based confidence: is #1 clearly ahead of #2, and is it supported?"""
    if not clusters:
        return {"level": "none", "margin": 0.0}
    top = clusters[0]
    second = clusters[1].score if len(clusters) > 1 else 0.0
    margin = top.score - second
    inl = top.best.inliers
    if inl >= 40 and top.support >= 2 and margin >= 1.0:
        level = "high"
    elif inl >= 20 and margin >= 0.5:
        level = "medium"
    else:
        level = "low"
    return {"level": level, "margin": round(margin, 3), "top_inliers": inl, "support_panos": top.support}


def save_ranking(clusters: list[Cluster], out: Path, top: int = 20) -> None:
    rows = []
    for i, cl in enumerate(clusters[:top], 1):
        rows.append({
            "rank": i, "lat": cl.lat, "lon": cl.lon, "heading": round(cl.heading, 1),
            "score": round(cl.score, 3), "support_panos": cl.support,
            "best": {k: v for k, v in asdict(cl.best).items() if k != "feats"},
            "features": cl.best.feats,
        })
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2))


# --- self-supervised training data --------------------------------------------------

def mine_training_rows(views: list, feats: np.ndarray, verifier, n_queries: int = 300, k: int = 50,
                       pos_radius_m: float = 30.0, seed: int = 0) -> list[dict]:
    """Use gallery views as pseudo-queries: positives are other panos within pos_radius_m
    looking roughly the same way (same facades), everything else is negative.

    TODO: query views come from the same capture as the gallery, so this underestimates
    real-world domain gap (phone vs. Street View camera, season). Mix in held-out older
    Street View dates or user photos with known coordinates when available.
    """
    from .geo import angle_diff
    from .retrieval import top_k

    rng = np.random.default_rng(seed)
    rows = []
    for qi in rng.choice(len(views), size=min(n_queries, len(views)), replace=False):
        q = views[qi]
        qf = verifier.query_features(q.path)
        for rank, (gi, sim) in enumerate(top_k(feats[qi], feats, k + 1)):
            g = views[gi]
            if g.pano_id == q.pano_id:
                continue
            v = verifier.verify(qf, g.path)
            label = int(dist_m((q.lat, q.lon), (g.lat, g.lon)) <= pos_radius_m
                        and angle_diff(q.heading, g.heading) <= 45)
            rows.append({"qid": int(qi), "label": label, "global_sim": sim, "global_rank": rank,
                         "inliers": v.inliers, "inlier_ratio": v.ratio,
                         # cluster features are approximated per pair at mining time
                         "cluster_max_inliers": v.inliers, "cluster_support": 1, "cluster_views": 1,
                         "heading_spread": 0.0})
    return rows


def train_lgbm(rows: list[dict], out: Path) -> None:
    import lightgbm as lgb

    rows = sorted(rows, key=lambda r: r["qid"])
    X = np.array([[r[k] for k in FEATURES] for r in rows], dtype=np.float32)
    y = np.array([r["label"] for r in rows])
    groups = np.unique([r["qid"] for r in rows], return_counts=True)[1]
    ds = lgb.Dataset(X, y, group=groups)
    params = {"objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [1, 5],
              "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 10, "verbose": -1}
    booster = lgb.train(params, ds, num_boost_round=300)
    booster.save_model(str(out))
