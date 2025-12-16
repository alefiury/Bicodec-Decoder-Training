from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from bicodec_train.sparktts.models.bicodec import BiCodec
from bicodec_train.sparktts.modules.encoder_decoder.wave_generator import WaveGenerator
from bicodec_train.sparktts.utils.file import load_config  # Spark-TTS util


def resolve_codec_dir(pretrained_dir: str) -> str:
    # Accept either .../SparkTTS-0.5B or .../SparkTTS-0.5B/BiCodec
    p = Path(pretrained_dir)

    if (p / "config.yaml").exists() and (p / "model.safetensors").exists():
        return str(p)
    if (p / "BiCodec" / "config.yaml").exists():
        return str(p / "BiCodec")
    raise FileNotFoundError(
        f"Could not find BiCodec checkpoint under '{pretrained_dir}'. Expected config.yaml and model.safetensors, or BiCodec/config.yaml."
    )


def load_bicodec(pretrained_dir: str, device: torch.device) -> Tuple[BiCodec, Dict[str, Any]]:
    codec_dir = resolve_codec_dir(pretrained_dir)
    cfg = load_config(f"{codec_dir}/config.yaml")
    # Spark-TTS stores audio tokenizer config under top-level key
    audio_cfg = cfg["audio_tokenizer"]
    model = BiCodec.load_from_checkpoint(codec_dir).to(device)
    model.eval()
    return model, audio_cfg


def build_decoder(audio_cfg: Dict[str, Any], rates: Optional[list[int]] = None, kernel_sizes: Optional[list[int]] = None) -> nn.Module:
    dec_cfg = dict(audio_cfg["decoder"])
    if rates is not None:
        dec_cfg["rates"] = rates
    if kernel_sizes is not None:
        dec_cfg["kernel_sizes"] = kernel_sizes
    return WaveGenerator(**dec_cfg)


def swap_decoder(
    model: BiCodec,
    audio_cfg: Dict[str, Any],
    rates: list[int],
    kernel_sizes: Optional[list[int]] = None,
    reuse_old_weights: bool = True,
) -> None:
    """Replace model.decoder with a new WaveGenerator.

    If reuse_old_weights=True, we load the old decoder state dict into the new decoder with strict=False,
    so all matching layers are reused.
    """
    old = model.decoder
    new_dec = build_decoder(audio_cfg, rates=rates, kernel_sizes=kernel_sizes)
    if reuse_old_weights:
        missing, unexpected = new_dec.load_state_dict(old.state_dict(), strict=False)
        # It's normal to miss keys when last upsampling block changes (16k -> 24k).
        if len(unexpected) > 0:
            print("[swap_decoder] unexpected keys:", unexpected)
        if len(missing) > 0:
            print("[swap_decoder] missing keys (expected for mismatched layers):", missing[:10], "..." if len(missing) > 10 else "")
    model.decoder = new_dec


def set_trainable(model: BiCodec, train_decoder_only: bool = True, train_prenet_postnet: bool = False) -> None:
    # default: freeze everything
    for p in model.parameters():
        p.requires_grad = False

    if train_prenet_postnet:
        for p in model.prenet.parameters():
            p.requires_grad = True
        for p in model.postnet.parameters():
            p.requires_grad = True

    if train_decoder_only:
        for p in model.decoder.parameters():
            p.requires_grad = True


@torch.no_grad()
def compute_d_vector(model: BiCodec, ref_wav_16k: torch.Tensor) -> torch.Tensor:
    """ref_wav_16k: (B, T) -> d_vector: (B, D)"""
    mel = model.mel_transformer(ref_wav_16k.unsqueeze(1)).squeeze(1)
    _, d_vector = model.speaker_encoder(mel.transpose(1, 2))
    return d_vector


def forward_decoder(
    model: BiCodec,
    feat: torch.Tensor,
    ref_wav_16k: Optional[torch.Tensor] = None,
    d_vector: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """feat: (B, frames, 1024)
    Either provide ref_wav_16k (B, Tref) or d_vector (B, D).
    Returns wav: (B, 1, T)
    """
    if d_vector is None:
        if ref_wav_16k is None:
            raise ValueError("Either ref_wav_16k or d_vector must be provided")
        d_vector = compute_d_vector(model, ref_wav_16k)

    z = model.encoder(feat.transpose(1, 2))
    vq = model.quantizer(z)
    x = model.prenet(vq["z_q"], d_vector)
    x = x + d_vector.unsqueeze(-1)
    wav = model.decoder(x)
    return wav
