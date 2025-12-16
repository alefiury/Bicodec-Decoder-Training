#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import soundfile as sf

from bicodec_train.models.bicodec_wrapper import load_bicodec, swap_decoder, forward_decoder
from bicodec_train.models.feat_extractor import Wav2Vec2MixFeat
from bicodec_train.utils.audio import read_audio_segment, resample_np, pad_or_trim_1d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_dir", type=str, required=True)
    p.add_argument("--audio", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--target_sr", type=int, default=24000)
    p.add_argument("--seconds", type=float, default=4.0)
    p.add_argument("--start_sec", type=float, default=0.0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--use_24k_decoder", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    codec, audio_cfg = load_bicodec(args.pretrained_dir, device=device)
    if args.use_24k_decoder:
        swap_decoder(codec, audio_cfg, rates=[8,5,4,3], reuse_old_weights=True)

    wav, sr = read_audio_segment(args.audio, start_sec=args.start_sec, duration_sec=args.seconds, mono=True)
    wav16 = resample_np(wav, sr, 16000)
    wav16 = torch.from_numpy(wav16).float().unsqueeze(0).to(device)
    ref = wav16[:, : min(wav16.shape[1], int(3.0 * 16000))]

    wav2vec2_dir = os.path.join(args.pretrained_dir, "wav2vec2-large-xlsr-53")
    feat_extractor = Wav2Vec2MixFeat(wav2vec2_dir=wav2vec2_dir, device=device)

    with torch.no_grad():
        feat = feat_extractor(wav16)
        y_hat = forward_decoder(codec, feat=feat, ref_wav_16k=ref)

    y_hat = y_hat.squeeze(0).squeeze(0).cpu().numpy()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out, y_hat, args.target_sr)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
