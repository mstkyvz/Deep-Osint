"""Command line interface.

    deeposint locate photo.jpg --area "Karamürsel, Kocaeli"
    deeposint locate photo.jpg --bbox 40.655,29.53,40.725,29.70 --verify sift --retrieval dinov2
    deeposint gallery --area "Karamürsel, Kocaeli" --work runs/karamursel
    deeposint clues photo.jpg --model qwen2.5vl:7b
    deeposint train-reranker --work runs/karamursel --out reranker.txt
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path


def _bbox(args) -> tuple[float, float, float, float]:
    from . import osm

    if args.bbox:
        s, w, n, e = (float(x) for x in args.bbox.split(","))
        return s, w, n, e
    if args.area:
        return osm.area_bbox(args.area)
    raise SystemExit("give --area or --bbox (south,west,north,east)")


def _box(s: str | None):
    return tuple(int(x) for x in s.split(",")) if s else None


def _add_area(p: argparse.ArgumentParser) -> None:
    p.add_argument("--area", help='place name resolved with Nominatim, e.g. "Karamürsel, Kocaeli"')
    p.add_argument("--bbox", help="south,west,north,east")
    p.add_argument("--work", default=None, help="working dir (cache, gallery, results)")
    p.add_argument("--step", type=float, default=40.0, help="sample spacing along roads, metres")
    p.add_argument("--residential", action="store_true", help="also scan residential/unclassified streets")


def _config(args):
    from . import osm
    from .pipeline import Config

    hw = osm.DEFAULT_HIGHWAYS + (("residential", "unclassified") if getattr(args, "residential", False) else ())
    slug = (args.area or args.bbox or "run").split(",")[0].strip().lower().replace(" ", "_")
    return Config(
        work_dir=Path(args.work or f"runs/{slug}"),
        step_m=args.step,
        highways=hw,
        retrieval_backend=getattr(args, "retrieval", "megaloc"),
        verify_backend=getattr(args, "verify", "lightglue"),
        device=getattr(args, "device", None),
        recall_k=getattr(args, "recall_k", 200),
        reranker_model=Path(args.reranker) if getattr(args, "reranker", None) else None,
        query_box=_box(getattr(args, "box", None)),
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="deeposint")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("locate", help="find where a street-level photo was taken")
    p.add_argument("photo")
    _add_area(p)
    p.add_argument("--retrieval", choices=["megaloc", "dinov2"], default="megaloc")
    p.add_argument("--verify", choices=["lightglue", "sift"], default="lightglue")
    p.add_argument("--recall-k", type=int, default=200)
    p.add_argument("--box", help="x0,y0,x1,y1 crop of the query to use (e.g. buildings only)")
    p.add_argument("--reranker", help="LightGBM model from train-reranker")
    p.add_argument("--device", help="cuda / mps / cpu")

    p = sub.add_parser("gallery", help="only build the Street View gallery for an area")
    _add_area(p)

    p = sub.add_parser("clues", help="extract clues with a local VLM via Ollama")
    p.add_argument("photo")
    p.add_argument("--model", default="qwen2.5vl:7b")

    p = sub.add_parser("train-reranker", help="mine self-supervised pairs from a gallery and train LightGBM")
    p.add_argument("--work", required=True)
    p.add_argument("--out", default="reranker.txt")
    p.add_argument("--queries", type=int, default=300)
    p.add_argument("--retrieval", choices=["megaloc", "dinov2"], default="megaloc")
    p.add_argument("--verify", choices=["lightglue", "sift"], default="lightglue")
    p.add_argument("--device")

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")

    if args.cmd == "locate":
        from .pipeline import locate

        print(json.dumps(locate(args.photo, _bbox(args), _config(args)), ensure_ascii=False, indent=2))
    elif args.cmd == "gallery":
        from .pipeline import build_gallery

        cfg = _config(args)
        cfg.work_dir.mkdir(parents=True, exist_ok=True)
        print(f"{len(build_gallery(_bbox(args), cfg))} views -> {cfg.work_dir / 'gallery'}")
    elif args.cmd == "clues":
        from .clues import extract

        print(json.dumps(extract(args.photo, args.model), ensure_ascii=False, indent=2))
    elif args.cmd == "train-reranker":
        from . import streetview
        from .rerank import mine_training_rows, train_lgbm
        from .retrieval import Embedder, build_index
        from .verify import Verifier

        work = Path(args.work)
        views = streetview.load_views(work / "gallery")
        emb = Embedder(args.retrieval, args.device)
        feats = build_index(emb, [v.path for v in views], work / f"index_{args.retrieval}.npz")
        rows = mine_training_rows(views, feats, Verifier(args.verify, args.device), n_queries=args.queries)
        (work / "rerank_rows.json").write_text(json.dumps(rows))
        train_lgbm(rows, Path(args.out))
        print(f"{len(rows)} pairs -> {args.out}")


if __name__ == "__main__":
    main()
