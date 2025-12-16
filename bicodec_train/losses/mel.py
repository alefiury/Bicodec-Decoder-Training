from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
import torchaudio


@dataclass
class MelConfig:
    sample_rate: int
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    f_min: float = 0.0
    f_max: Optional[float] = None


class MelL1Loss(torch.nn.Module):
    def __init__(self, cfg: MelConfig):
        super().__init__()
        self.cfg = cfg
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            win_length=cfg.win_length,
            n_mels=cfg.n_mels,
            f_min=cfg.f_min,
            f_max=cfg.f_max,
            power=1.0,
        )

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # y_hat,y: (B, T)
        m_hat = self.mel(y_hat)
        m = self.mel(y)
        return F.l1_loss(torch.log(m_hat + 1e-6), torch.log(m + 1e-6))
