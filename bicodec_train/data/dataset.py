from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

import numpy as np
import torch
from torch.utils.data import Dataset

from .manifest import ManifestItem
from .cache import load_cache
from ..utils.audio import read_audio_segment, resample_np, pad_or_trim_1d


@dataclass
class SegmentConfig:
    fps: int = 50  # frames per second (wav2vec2 conv stride @ 16k: 320 samples -> 50 fps)
    segment_seconds: float = 2.56
    ref_seconds: float = 3.0
    align_to_frames: bool = True


class BiCodecDecoderDataset(Dataset):
    def __init__(
        self,
        items: List[ManifestItem],
        target_sample_rate: int,
        encoder_sample_rate: int = 16000,
        segment: SegmentConfig = SegmentConfig(),
        cache_dir: Optional[str] = None,
        use_cached_dvector: bool = False,
    ) -> None:
        super().__init__()
        self.items = items
        self.target_sr = int(target_sample_rate)
        self.enc_sr = int(encoder_sample_rate)
        self.segment = segment
        self.cache_dir = cache_dir
        self.use_cached_dvector = use_cached_dvector

        # segment in frames (must be integer)
        self.seg_frames = max(1, int(round(self.segment.segment_seconds * self.segment.fps)))
        self.seg_seconds = self.seg_frames / self.segment.fps

        # reference segment
        self.ref_frames = max(1, int(round(self.segment.ref_seconds * self.segment.fps)))
        self.ref_seconds = self.ref_frames / self.segment.fps

        # expected sample lengths for perfect alignment
        self.target_hop = self.target_sr // self.segment.fps
        self.enc_hop = self.enc_sr // self.segment.fps
        if self.target_hop * self.segment.fps != self.target_sr:
            raise ValueError(f"target_sample_rate={self.target_sr} must be divisible by fps={self.segment.fps}")
        if self.enc_hop * self.segment.fps != self.enc_sr:
            raise ValueError(f"encoder_sample_rate={self.enc_sr} must be divisible by fps={self.segment.fps}")

        self.seg_len_target = self.seg_frames * self.target_hop
        self.seg_len_enc = self.seg_frames * self.enc_hop
        self.ref_len_enc = self.ref_frames * self.enc_hop

    def __len__(self) -> int:
        return len(self.items)

    def _choose_start_frame(self, total_frames: int) -> int:
        if total_frames <= self.seg_frames + 1:
            return 0
        return random.randint(0, total_frames - self.seg_frames - 1)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        total_frames = max(1, int(item.duration_sec * self.segment.fps))

        start_frame = self._choose_start_frame(total_frames)
        start_sec = start_frame / self.segment.fps

        # Read target segment (native)
        wav_seg, src_sr = read_audio_segment(item.audio_path, start_sec=start_sec, duration_sec=self.seg_seconds, mono=True)
        wav_target = resample_np(wav_seg, src_sr, self.target_sr)
        wav_target = pad_or_trim_1d(wav_target, self.seg_len_target)

        # Read encoder segment (16k) from the same time
        wav_enc = resample_np(wav_seg, src_sr, self.enc_sr)
        wav_enc = pad_or_trim_1d(wav_enc, self.seg_len_enc)

        # Reference for speaker condition: either a random chunk or reuse same segment
        # Choose reference aligned to frames too.
        ref_start_frame = self._choose_start_frame(total_frames) if total_frames > self.ref_frames else 0
        ref_start_sec = ref_start_frame / self.segment.fps
        wav_ref_seg, _ = read_audio_segment(item.audio_path, start_sec=ref_start_sec, duration_sec=self.ref_seconds, mono=True)
        wav_ref = resample_np(wav_ref_seg, src_sr, self.enc_sr)
        wav_ref = pad_or_trim_1d(wav_ref, self.ref_len_enc)

        out: Dict[str, Any] = {
            "audio_path": item.audio_path,
            "start_frame": start_frame,
            "seg_frames": self.seg_frames,
            "wav_target": torch.from_numpy(wav_target).float(),  # (T_target,)
            "wav_enc": torch.from_numpy(wav_enc).float(),        # (T_enc,)
            "ref_wav": torch.from_numpy(wav_ref).float(),        # (T_ref_enc,)
        }

        if self.cache_dir is not None:
            cached = load_cache(self.cache_dir, item.audio_path)
            if cached is not None and "feat" in cached:
                feat = cached["feat"]  # (T_total, 1024)
                # Slice frames; guard for short clips
                start = min(start_frame, max(0, feat.shape[0] - self.seg_frames))
                out["feat"] = feat[start : start + self.seg_frames].clone()
            if self.use_cached_dvector and cached is not None and "d_vector" in cached:
                out["d_vector"] = cached["d_vector"].clone()

        return out


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Padless because we use fixed-length aligned segments
    keys = batch[0].keys()
    out: Dict[str, Any] = {}
    for k in keys:
        if k in ("audio_path",):
            out[k] = [b[k] for b in batch]
        elif isinstance(batch[0][k], torch.Tensor):
            out[k] = torch.stack([b[k] for b in batch], dim=0)
        else:
            out[k] = [b[k] for b in batch]
    return out
