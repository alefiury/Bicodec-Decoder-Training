from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn
import torch.nn.functional as F


def discriminator_loss(d_real: List[torch.Tensor], d_fake: List[torch.Tensor]) -> torch.Tensor:
    loss = 0.0
    for dr, df in zip(d_real, d_fake):
        loss = loss + torch.mean((1.0 - dr) ** 2) + torch.mean(df ** 2)
    return loss


def generator_adversarial_loss(d_fake: List[torch.Tensor]) -> torch.Tensor:
    loss = 0.0
    for df in d_fake:
        loss = loss + torch.mean((1.0 - df) ** 2)
    return loss


def feature_matching_loss(fmaps_real: List[List[torch.Tensor]], fmaps_fake: List[List[torch.Tensor]]) -> torch.Tensor:
    loss = 0.0
    n = 0
    for fr, ff in zip(fmaps_real, fmaps_fake):
        for a, b in zip(fr, ff):
            loss = loss + F.l1_loss(b, a.detach())
            n += 1
    return loss / max(n, 1)


class DiscriminatorS(nn.Module):
    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()
        norm_f = nn.utils.spectral_norm if use_spectral_norm else nn.utils.weight_norm
        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(nn.Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(nn.Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return x, fmap


class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(use_spectral_norm=False),
            DiscriminatorS(use_spectral_norm=False),
        ])
        self.avgpools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2),
        ])

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        outs = []
        fmaps = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                x = self.avgpools[i - 1](x)
            o, fm = d(x)
            outs.append(o)
            fmaps.append(fm)
        return outs, fmaps


class DiscriminatorP(nn.Module):
    def __init__(self, period: int):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv2d(1, 32, (5, 1), (3, 1), padding=(2, 0))),
            nn.utils.weight_norm(nn.Conv2d(32, 128, (5, 1), (3, 1), padding=(2, 0))),
            nn.utils.weight_norm(nn.Conv2d(128, 512, (5, 1), (3, 1), padding=(2, 0))),
            nn.utils.weight_norm(nn.Conv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0))),
            nn.utils.weight_norm(nn.Conv2d(1024, 1024, (5, 1), (1, 1), padding=(2, 0))),
        ])
        self.conv_post = nn.utils.weight_norm(nn.Conv2d(1024, 1, (3, 1), (1, 1), padding=(1, 0)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # x: (B,1,T)
        b, c, t = x.shape
        if t % self.period != 0:
            pad = self.period - (t % self.period)
            x = F.pad(x, (0, pad), mode="reflect")
            t = t + pad
        x = x.view(b, c, t // self.period, self.period)  # (B,1,T/P,P)

        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods: List[int] = [2, 3, 5, 7, 11]):
        super().__init__()
        self.discriminators = nn.ModuleList([DiscriminatorP(p) for p in periods])

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        outs = []
        fmaps = []
        for d in self.discriminators:
            o, fm = d(x)
            outs.append(o)
            fmaps.append(fm)
        return outs, fmaps


class HiFiGANDiscriminators(nn.Module):
    def __init__(self):
        super().__init__()
        self.msd = MultiScaleDiscriminator()
        self.mpd = MultiPeriodDiscriminator()

    def forward(self, x: torch.Tensor):
        # returns all outs and fmaps
        outs_s, fmaps_s = self.msd(x)
        outs_p, fmaps_p = self.mpd(x)
        return (outs_s + outs_p), (fmaps_s + fmaps_p)
