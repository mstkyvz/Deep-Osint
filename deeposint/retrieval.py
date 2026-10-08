"""Global image descriptors for coarse retrieval (recall stage).

Backends:
- ``megaloc``: MegaLoc (Berton & Masone, 2025), a visual place recognition model
  trained to answer "is this the same place?" — preferred.
- ``dinov2``: generic DINOv2 ViT-B/14 CLS token — fallback, worse under season /
  lighting change, but no extra repo.

Both load through ``torch.hub`` on first use (downloads weights to ~/.cache/torch).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from .imageio import open_rgb

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def pick_device(pref: str | None = None) -> str:
    import torch

    if pref:
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Embedder:
    def __init__(self, backend: str = "megaloc", device: str | None = None, size: int = 322):
        import torch

        self.backend = backend
        self.device = pick_device(device)
        self.size = size - size % 14  # ViT patch size
        if backend == "megaloc":
            # TODO(verify): entry point name per https://github.com/gmberton/MegaLoc README.
            self.model = torch.hub.load("gmberton/MegaLoc", "get_trained_model")
        elif backend == "dinov2":
            self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        else:
            raise ValueError(f"unknown backend {backend}")
        self.model = self.model.eval().to(self.device)

    def _tensor(self, imgs: list[Image.Image]):
        import torch

        arr = np.stack([
            np.asarray(im.convert("RGB").resize((self.size, self.size), Image.BICUBIC), dtype=np.float32) / 255.0
            for im in imgs
        ])
        arr = (arr - np.array(_MEAN, dtype=np.float32)) / np.array(_STD, dtype=np.float32)
        return torch.from_numpy(arr).permute(0, 3, 1, 2).to(self.device)

    def embed(self, imgs: list[Image.Image]) -> np.ndarray:
        import torch

        with torch.inference_mode():
            out = self.model(self._tensor(imgs))
        if isinstance(out, (tuple, list)):
            out = out[0]
        feats = out.float().cpu().numpy()
        return feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)

    def embed_query(self, path: str | Path, box: tuple[int, int, int, int] | None = None) -> np.ndarray:
        """Query descriptor; averaged over the full frame and a few crops for robustness to framing."""
        im = open_rgb(path)
        if box:
            im = im.crop(box)
        w, h = im.size
        crops = [im,
                 im.crop((0, 0, int(w * 0.8), h)),
                 im.crop((int(w * 0.2), 0, w, h)),
                 im.crop((0, int(h * 0.15), w, int(h * 0.85)))]
        f = self.embed(crops).mean(axis=0)
        return f / (np.linalg.norm(f) + 1e-8)


def build_index(embedder: Embedder, paths: list[str], out: Path, batch: int = 32) -> np.ndarray:
    """Embed every gallery image; cached as .npz next to the gallery."""
    if out.exists():
        z = np.load(out, allow_pickle=True)
        if list(z["paths"]) == list(paths) and str(z["backend"]) == embedder.backend:
            return z["feats"]
    feats = []
    for i in tqdm(range(0, len(paths), batch), desc=f"embed ({embedder.backend})"):
        feats.append(embedder.embed([open_rgb(p) for p in paths[i:i + batch]]))
    arr = np.concatenate(feats) if feats else np.zeros((0, 1), np.float32)
    np.savez(out, feats=arr, paths=np.array(paths), backend=embedder.backend)
    return arr


def top_k(query: np.ndarray, gallery: np.ndarray, k: int) -> list[tuple[int, float]]:
    sims = gallery @ query
    k = min(k, len(sims))
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return [(int(i), float(sims[i])) for i in idx]
