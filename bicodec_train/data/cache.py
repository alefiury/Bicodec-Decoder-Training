from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, Dict, Any

import torch


def key_for_path(audio_path: str) -> str:
    return hashlib.sha1(audio_path.encode("utf-8")).hexdigest()


def cache_path(cache_dir: str, audio_path: str) -> Path:
    return Path(cache_dir) / f"{key_for_path(audio_path)}.pt"


def load_cache(cache_dir: str, audio_path: str) -> Optional[Dict[str, Any]]:
    p = cache_path(cache_dir, audio_path)
    if not p.exists():
        return None
    return torch.load(p, map_location="cpu")


def save_cache(cache_dir: str, audio_path: str, data: Dict[str, Any]) -> None:
    p = cache_path(cache_dir, audio_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, p)
