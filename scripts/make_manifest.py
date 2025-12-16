#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List

import soundfile as sf
from tqdm import tqdm


def iter_audio_files(root: str, exts: List[str]) -> Iterable[str]:
    exts = [e.lower().lstrip(".") for e in exts]
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().split(".")[-1] in exts:
                yield str(Path(dirpath) / fn)


def duration_sec(path: str) -> float:
    with sf.SoundFile(path) as f:
        return float(len(f)) / float(f.samplerate)


def make_manifest(files: List[str], args: argparse.Namespace):
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for ap in tqdm(files, desc="manifest"):
            try:
                dur = duration_sec(ap)
            except Exception as e:
                print(f"[skip] {ap}: {e}")
                continue
            obj = {"audio_path": os.path.abspath(ap), "duration_sec": dur}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Wrote {out_path} with {len(files)} entries (skips possible).")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--extensions", nargs="+", default=["wav", "flac", "mp3", "ogg", "m4a"])
    args = p.parse_args()

    files = list(iter_audio_files(args.input_dir, args.extensions))
    if len(files) == 0:
        raise SystemExit("No audio files found.")

    # split train and val
    train_files = files[:int(len(files) * 0.8)]
    val_files = files[int(len(files) * 0.8):]

    make_manifest(train_files, argparse.Namespace(output=args.output.replace(".jsonl", "_train.jsonl")))
    make_manifest(val_files, argparse.Namespace(output=args.output.replace(".jsonl", "_val.jsonl")))


if __name__ == "__main__":
    main()
