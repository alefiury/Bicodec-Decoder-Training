from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any

import torch
import torch.nn.functional as F


@dataclass
class STFTConfig:
    fft_size: int
    hop_size: int
    win_length: int


def _stft_mag(x: torch.Tensor, cfg: STFTConfig) -> torch.Tensor:
    # x: (B, T)
    window = torch.hann_window(cfg.win_length, device=x.device)
    spec = torch.stft(
        x,
        n_fft=cfg.fft_size,
        hop_length=cfg.hop_size,
        win_length=cfg.win_length,
        window=window,
        center=True,
        return_complex=True,
    )
    mag = spec.abs()
    return mag


class MultiResolutionSTFTLoss(torch.nn.Module):
    def __init__(self, configs: List[Dict[str, Any]]):
        super().__init__()
        self.configs = [STFTConfig(**c) for c in configs]

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # y_hat, y: (B, T)
        loss = 0.0
        for cfg in self.configs:
            mag_hat = _stft_mag(y_hat, cfg)
            mag = _stft_mag(y, cfg)
            sc = torch.norm(mag - mag_hat, p="fro") / (torch.norm(mag, p="fro") + 1e-9)
            mag_loss = F.l1_loss(torch.log(mag_hat + 1e-7), torch.log(mag + 1e-7))
            loss = loss + sc + mag_loss
        return loss / len(self.configs)
