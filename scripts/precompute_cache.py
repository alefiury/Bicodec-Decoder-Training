#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Any

import torch
from tqdm import tqdm

from bicodec_train.data.manifest import load_jsonl
from bicodec_train.data.cache import save_cache
from bicodec_train.utils.audio import read_audio_segment, resample_np
from bicodec_train.models.bicodec_wrapper import load_bicodec, compute_d_vector
from bicodec_train.models.feat_extractor import Wav2Vec2MixFeat


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=str, required=True)
    p.add_argument("--cache_dir", type=str, required=True)
    p.add_argument("--pretrained_dir", type=str, required=True)
    p.add_argument("--encoder_sr", type=int, default=16000)
    p.add_argument("--ref_seconds", type=float, default=3.0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--cache_dvector", action="store_true", help="Also cache speaker condition d_vector.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    items = load_jsonl(args.manifest)

    codec, _audio_cfg = load_bicodec(args.pretrained_dir, device=device)
    wav2vec2_dir = os.path.join(args.pretrained_dir, "wav2vec2-large-xlsr-53")
    feat_extractor = Wav2Vec2MixFeat(wav2vec2_dir=wav2vec2_dir, device=device)

    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    for it in tqdm(items, desc="cache"):
        # full audio -> 16k for wav2vec2
        wav_full, src_sr = read_audio_segment(it.audio_path, start_sec=0.0, duration_sec=it.duration_sec, mono=True)
        wav16 = resample_np(wav_full, src_sr, args.encoder_sr)
        wav16_t = torch.from_numpy(wav16).float().unsqueeze(0).to(device)

        with torch.no_grad():
            feat = feat_extractor(wav16_t).squeeze(0).cpu()  # (frames, 1024)

        cache_obj: Dict[str, Any] = {"feat": feat}

        if args.cache_dvector:
            # take first ref_seconds from 16k audio
            ref_len = int(args.ref_seconds * args.encoder_sr)
            ref = torch.from_numpy(wav16[:ref_len]).float().unsqueeze(0).to(device)
            with torch.no_grad():
                dvec = compute_d_vector(codec, ref).squeeze(0).cpu()
            cache_obj["d_vector"] = dvec

        save_cache(args.cache_dir, it.audio_path, cache_obj)

    print(f"Cache written to {args.cache_dir}")


if __name__ == "__main__":
    main()
