# BiCodec Decoder Training (16 kHz and 24 kHz)

This repository provides a **decoder-only** training pipeline for Spark-TTS **BiCodec**:

- Train / fine-tune the **16 kHz decoder** (original BiCodec).
- Train / fine-tune a **24 kHz decoder** by **keeping the encoder + quantizer + speaker encoder unchanged**
  and only changing the **waveform generator upsampling** from `320 (= 16k/50)` to `480 (= 24k/50)`.

The intended usage is:
- start from a **pretrained Spark-TTS BiCodec checkpoint directory** (local files, no internet),
- optionally **reuse** the 16 kHz decoder weights for all matching layers,
- train a new decoder that outputs **24 kHz** audio.

## 0) Assumptions

You have a local Spark-TTS pretrained model directory like:

```
pretrained_models/SparkTTS-0.5B/
  BiCodec/
    config.yaml
    model.safetensors
  wav2vec2-large-xlsr-53/   # local HF files
```

This trainer uses the same wav2vec2 feature extraction as Spark-TTS:
- audio is **resampled to 16 kHz** for wav2vec2 features,
- the decoder is trained to reconstruct **target audio** at 16 kHz or 24 kHz.

> If your dataset is truly native 24 kHz, the 24 kHz decoder learns a bandwidth-extension / upsampling prior
> conditioned on the same 50-tps features.

## 1) Install

Create a venv and install:

```bash
pip install -r requirements.txt
```

Install Spark-TTS **editable** (so we can import `sparktts.*`):

```bash
pip install -e /path/to/Spark-TTS-main
```

## 2) Prepare a manifest

Make a JSONL manifest from a folder of audio files:

```bash
python scripts/make_manifest.py \
  --input_dir /path/to/wavs \
  --output data/train.jsonl \
  --extensions wav flac mp3
```

Split into train/val however you want (or just point val to a smaller file).

Each line is like:

```json
{"audio_path": "/abs/path/file.wav", "duration_sec": 4.83}
```

## 3) (Optional) Precompute features cache (recommended)

This caches wav2vec2 features (50 fps) and an optional speaker condition vector.

```bash
python scripts/precompute_cache.py \
  --manifest data/train.jsonl \
  --cache_dir cache/train \
  --pretrained_dir /path/to/pretrained_models/SparkTTS-0.5B \
  --num_workers 8
```

Do the same for val.

## 4) Train 24 kHz decoder (decoder-only)

```bash
python -m bicodec_train.train \
  --config configs/train_24k.yaml \
  model.pretrained_dir=/path/to/pretrained_models/SparkTTS-0.5B \
  data.train_manifest=data/train.jsonl \
  data.val_manifest=data/val.jsonl \
  data.cache.dir=cache/train \
  data.val_cache.dir=cache/val \
  exp.out_dir=outputs/decoder24k
```

### Reuse weights from a 16 kHz decoder checkpoint

If your `pretrained_dir` is 16 kHz BiCodec (original), the trainer will:
- build a 24 kHz WaveGenerator with rates `[8,5,4,3]`,
- load the 16 kHz decoder state dict **with `strict=False`** so matching layers are reused,
- re-init the new last upsampling block (shape mismatch).

This behavior is controlled by:

```
model.init.reuse_16k_decoder_weights: true
```

## 5) Fine-tune a 16 kHz decoder (decoder-only)

```bash
python -m bicodec_train.train \
  --config configs/train_16k_finetune.yaml \
  model.pretrained_dir=/path/to/pretrained_models/SparkTTS-0.5B \
  data.train_manifest=data/train.jsonl \
  data.val_manifest=data/val.jsonl \
  exp.out_dir=outputs/decoder16k_ft
```

## Notes on losses

Default config uses a **non-adversarial** set of losses (fast + stable):
- multi-resolution STFT
- mel L1
- waveform L1

You can enable a HiFi-GAN style discriminator with feature matching:
`loss.use_gan: true` (see configs).

---

## Troubleshooting

- If the saved audio plays at the wrong speed, check that you are writing with the correct sample rate.
  This trainer writes reconstructions using `data.target_sample_rate`.
- If you get off-by-one length mismatches, set `data.align_to_frames: true` (default) and use frame-aligned starts.

