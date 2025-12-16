from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
from omegaconf import OmegaConf


def load_config(path: str, overrides: Optional[list[str]] = None):
    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    OmegaConf.set_struct(cfg, False)
    return cfg


def to_dict(cfg) -> Dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore
