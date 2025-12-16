from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model


class Wav2Vec2MixFeat(nn.Module):
    """Replicates Spark-TTS wav2vec2 feature extraction:
    feats = (hidden[11] + hidden[14] + hidden[16]) / 3
    """

    def __init__(self, wav2vec2_dir: str, device: torch.device):
        super().__init__()
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec2_dir)
        self.model = Wav2Vec2Model.from_pretrained(wav2vec2_dir).to(device)
        self.model.config.output_hidden_states = True
        self.device = device
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, wav_16k: torch.Tensor) -> torch.Tensor:
        """wav_16k: (B, T) float32 in [-1, 1]
        returns: (B, frames, 1024)
        """
        # HF processor accepts list/np arrays or torch; we follow Spark-TTS.
        if isinstance(wav_16k, torch.Tensor):
            wav_list = [w.detach().cpu().numpy() for w in wav_16k]
        else:
            wav_list = wav_16k
        inputs = self.processor(
            wav_list,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            output_hidden_states=True,
        ).input_values
        out = self.model(inputs.to(self.device))
        feats = (out.hidden_states[11] + out.hidden_states[14] + out.hidden_states[16]) / 3.0
        return feats
