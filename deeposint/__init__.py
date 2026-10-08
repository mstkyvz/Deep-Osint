"""Deep-Osint: street-level photo geolocation without a large LLM in the loop.

Pipeline (see README):
    area -> OSM roads -> Street View gallery -> global retrieval (MegaLoc / DINOv2)
         -> local verification (LightGlue / SIFT + RANSAC) -> neighbour voting
         -> re-ranking (heuristic or LightGBM) -> coordinate + confidence
"""

__version__ = "0.1.0"
