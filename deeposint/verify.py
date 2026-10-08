"""Local feature verification (precision stage).

Counts geometrically consistent correspondences between the query and a
candidate view: keypoints + matcher, then a RANSAC fundamental matrix.
Repetitive textures (kerbs, road markings, sky) give many raw matches but few
RANSAC inliers, which is exactly the failure we saw with plain SIFT ranking.

Backends:
- ``lightglue``: SuperPoint + LightGlue (cvg/LightGlue) — preferred.
- ``sift``: OpenCV SIFT + ratio test — no extra deps, weaker.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .imageio import open_rgb


@dataclass
class Verification:
    matches: int
    inliers: int

    @property
    def ratio(self) -> float:
        return self.inliers / self.matches if self.matches else 0.0


def _ransac_inliers(k0: np.ndarray, k1: np.ndarray) -> int:
    import cv2

    if len(k0) < 8:
        return 0
    _, mask = cv2.findFundamentalMat(k0, k1, cv2.USAC_MAGSAC, 3.0, 0.999, 10000)
    return int(mask.sum()) if mask is not None else 0


def _mask_lower(im: Image.Image, frac: float) -> Image.Image:
    """Grey out the bottom `frac` of the frame (road surface / kerbs) before matching."""
    if frac <= 0:
        return im
    im = im.copy()
    w, h = im.size
    im.paste((128, 128, 128), (0, int(h * (1 - frac)), w, h))
    return im


class Verifier:
    def __init__(self, backend: str = "lightglue", device: str | None = None, max_side: int = 1024,
                 mask_road: float = 0.25, max_keypoints: int = 2048):
        self.backend = backend
        self.max_side = max_side
        self.mask_road = mask_road
        if backend == "lightglue":
            from lightglue import LightGlue, SuperPoint

            from .retrieval import pick_device

            self.device = pick_device(device)
            self.extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(self.device)
            self.matcher = LightGlue(features="superpoint").eval().to(self.device)
        elif backend == "sift":
            import cv2

            self.sift = cv2.SIFT_create(nfeatures=4000)
            self.bf = cv2.BFMatcher(cv2.NORM_L2)
        else:
            raise ValueError(f"unknown backend {backend}")
        self._qcache: dict[str, object] = {}

    def _prep(self, im: Image.Image) -> Image.Image:
        im = _mask_lower(im, self.mask_road)
        w, h = im.size
        s = self.max_side / max(w, h)
        return im.resize((int(w * s), int(h * s)), Image.BILINEAR) if s < 1 else im

    # --- LightGlue -----------------------------------------------------------------
    def _lg_feats(self, im: Image.Image):
        import torch

        t = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1).to(self.device)
        with torch.inference_mode():
            return self.extractor.extract(t)

    def _lg_match(self, f0, f1) -> Verification:
        import torch
        from lightglue.utils import rbd

        with torch.inference_mode():
            m = self.matcher({"image0": f0, "image1": f1})
        a, b, m = rbd(f0), rbd(f1), rbd(m)
        idx = m["matches"].cpu().numpy()
        k0 = a["keypoints"].cpu().numpy()[idx[:, 0]]
        k1 = b["keypoints"].cpu().numpy()[idx[:, 1]]
        return Verification(len(idx), _ransac_inliers(k0, k1))

    # --- SIFT ----------------------------------------------------------------------
    def _sift_feats(self, im: Image.Image):
        import cv2

        g = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2GRAY)
        return self.sift.detectAndCompute(g, None)

    def _sift_match(self, f0, f1) -> Verification:
        (kp0, d0), (kp1, d1) = f0, f1
        if d0 is None or d1 is None or len(kp0) < 8 or len(kp1) < 8:
            return Verification(0, 0)
        good = [m for m, n in self.bf.knnMatch(d0, d1, k=2) if m.distance < 0.8 * n.distance]
        k0 = np.float32([kp0[m.queryIdx].pt for m in good])
        k1 = np.float32([kp1[m.trainIdx].pt for m in good])
        return Verification(len(good), _ransac_inliers(k0, k1))

    # --- public --------------------------------------------------------------------
    def query_features(self, path: str, box: tuple[int, int, int, int] | None = None):
        key = f"{path}|{box}"
        if key not in self._qcache:
            im = open_rgb(path)
            if box:
                im = im.crop(box)
            im = self._prep(im)
            self._qcache[key] = self._lg_feats(im) if self.backend == "lightglue" else self._sift_feats(im)
        return self._qcache[key]

    def verify(self, query_feats, candidate_path: str) -> Verification:
        im = self._prep(open_rgb(candidate_path))
        if self.backend == "lightglue":
            return self._lg_match(query_feats, self._lg_feats(im))
        return self._sift_match(query_feats, self._sift_feats(im))
