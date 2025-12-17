from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass
class ManifestItem:
    audio_path: str
    duration_sec: float


def load_jsonl(path: str) -> List[ManifestItem]:
    items: List[ManifestItem] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            items.append(
                ManifestItem(
                    audio_path=obj["audio_path"],
                    duration_sec=float(obj["duration_sec"]),
                )
            )
    return items
