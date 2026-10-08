"""Optional clue extraction with a small local vision-language model (Ollama).

The VLM is *not* the decision maker. It only fills a fixed JSON schema that the
pipeline uses for cheap routing (e.g. skip the street pipeline for a mountain
photo, read a shop sign to seed an OSM name search). Recommended models:
``qwen2.5vl:7b`` or ``gemma3:12b`` (4-bit, ~6-8 GB RAM).
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

SCHEMA = {
    "type": "object",
    "properties": {
        "scene": {"type": "string", "enum": ["urban", "suburban", "rural", "coastal", "mountain", "indoor", "other"]},
        "road_type": {"type": "string", "enum": ["divided_highway", "boulevard", "street", "alley", "none"]},
        "has_water": {"type": "boolean"},
        "has_mountain_skyline": {"type": "boolean"},
        "landmark": {"type": "string", "description": "named landmark if confidently recognised, else empty"},
        "texts": {"type": "array", "items": {"type": "string"}, "description": "verbatim legible text: signs, shop names, plates"},
        "country_guess": {"type": "string"},
        "notes": {"type": "string"},
    },
    "required": ["scene", "road_type", "has_water", "has_mountain_skyline", "landmark", "texts", "country_guess"],
}

PROMPT = (
    "You are a geolocation assistant. Describe ONLY what is visible in the photo and fill the JSON schema. "
    "Copy text exactly as written; do not translate or guess unreadable characters (use '?'). "
    "Leave 'landmark' empty unless you are confident."
)


def extract(photo: str | Path, model: str = "qwen2.5vl:7b", timeout: float = 180) -> dict:
    img_b64 = base64.b64encode(Path(photo).read_bytes()).decode()
    body = {
        "model": model,
        "stream": False,
        "format": SCHEMA,
        "options": {"temperature": 0},
        "messages": [{"role": "user", "content": PROMPT, "images": [img_b64]}],
    }
    r = requests.post(f"{OLLAMA_URL}/api/chat", json=body, timeout=timeout)
    r.raise_for_status()
    return json.loads(r.json()["message"]["content"])
