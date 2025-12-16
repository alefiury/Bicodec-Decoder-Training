from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio


def read_audio_segment(
    path: str,
    start_sec: float,
    duration_sec: float,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    """Read a segment from an audio file using soundfile (seeking by frame).

    Returns:
      audio: float32 numpy array shape (T,) if mono else (T, C)
      sr: sample rate
    """
    with sf.SoundFile(path) as f:
        sr = f.samplerate
        start_frame = int(round(start_sec * sr))
        num_frames = int(round(duration_sec * sr))
        start_frame = max(0, min(start_frame, len(f) - 1))
        f.seek(start_frame)
        audio = f.read(frames=num_frames, dtype="float32", always_2d=not mono)
    if mono:
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
    else:
        # soundfile returns (T, C) when always_2d=True
        pass
    return audio.astype(np.float32), sr


def resample_np(audio: np.ndarray, orig_sr: int, new_sr: int) -> np.ndarray:
    if orig_sr == new_sr:
        return audio
    x = torch.from_numpy(audio).float().unsqueeze(0)  # (1, T)
    resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=new_sr)
    y = resampler(x).squeeze(0).cpu().numpy()
    return y.astype(np.float32)


def pad_or_trim_1d(x: np.ndarray, length: int) -> np.ndarray:
    if x.shape[0] == length:
        return x
    if x.shape[0] > length:
        return x[:length]
    pad = np.zeros((length - x.shape[0],), dtype=x.dtype)
    return np.concatenate([x, pad], axis=0)
