from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import wandb
import torch.nn.functional as F
from torch.utils.data import DataLoader

import lightning as L
from lightning.pytorch.utilities import grad_norm
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.optim import AdamW


from omegaconf import OmegaConf

from .config import load_config, to_dict
from .data.manifest import load_jsonl
from .data.dataset import BiCodecDecoderDataset, SegmentConfig, collate_fn
from .models.bicodec_wrapper import load_bicodec, swap_decoder, set_trainable, forward_decoder
from .models.feat_extractor import Wav2Vec2MixFeat
from .losses.stft import MultiResolutionSTFTLoss
from .losses.mel import MelL1Loss, MelConfig
from .losses.gan import (
    HiFiGANDiscriminators,
    discriminator_loss,
    generator_adversarial_loss,
    feature_matching_loss,
)
from .utils.schedulers import CosineWarmupLR

import soundfile as sf


class DecoderTrainModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))

        self.cfg = cfg
        self.target_sr = int(cfg.data.target_sample_rate)
        self.enc_sr = int(cfg.data.encoder_sample_rate)

        device = torch.device("cuda" if torch.cuda.is_available() and cfg.model.device == "cuda" else "cpu")
        self.codec, self.audio_cfg = load_bicodec(cfg.model.pretrained_dir, device=device)

        # Optional decoder override (e.g. 24 kHz)
        if cfg.model.decoder_override.enabled:
            swap_decoder(
                self.codec,
                self.audio_cfg,
                rates=list(cfg.model.decoder_override.rates),
                kernel_sizes=list(cfg.model.decoder_override.kernel_sizes) if cfg.model.decoder_override.kernel_sizes is not None else None,
                reuse_old_weights=bool(cfg.model.init.reuse_16k_decoder_weights),
            )

        set_trainable(
            self.codec,
            train_decoder_only=bool(cfg.model.train.decoder_only),
            train_prenet_postnet=bool(cfg.model.train.prenet_postnet),
        )

        # wav2vec2 feature extractor (frozen)
        wav2vec2_dir = os.path.join(cfg.model.pretrained_dir, "wav2vec2-large-xlsr-53")
        self.feat_extractor = Wav2Vec2MixFeat(wav2vec2_dir=wav2vec2_dir, device=device)

        # losses
        self.stft_loss = MultiResolutionSTFTLoss(configs=to_dict(cfg.loss.stft.configs))
        mel_cfg = MelConfig(
            sample_rate=self.target_sr,
            n_fft=int(cfg.loss.mel.n_fft),
            hop_length=int(cfg.loss.mel.hop_length),
            win_length=int(cfg.loss.mel.win_length),
            n_mels=int(cfg.loss.mel.n_mels),
            f_min=float(cfg.loss.mel.f_min),
            f_max=float(cfg.loss.mel.f_max) if cfg.loss.mel.f_max is not None else None,
        )
        self.mel_loss = MelL1Loss(mel_cfg)

        self.use_gan = bool(cfg.loss.use_gan)
        if self.use_gan:
            self.disc = HiFiGANDiscriminators()
            self.automatic_optimization = False

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset, self.val_dataset = build_dataset(self.cfg)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=int(self.cfg.train.batch_size),
            shuffle=True,
            num_workers=int(self.cfg.train.num_workers),
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
            persistent_workers=self.cfg.train.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=int(self.cfg.train.val_batch_size),
            shuffle=False,
            num_workers=int(self.cfg.train.num_workers),
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
            persistent_workers=self.cfg.train.num_workers > 0,
        )

    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        dataset = self.train_dataloader()
        if self.trainer.max_steps and self.trainer.max_steps > 0:
            return self.trainer.max_steps
        dataset_size = len(dataset)

        gpu_count = self.trainer.num_devices if self.trainer.num_devices else 1
        accumulate_grad_batches = self.trainer.accumulate_grad_batches

        effective_batches = dataset_size // (gpu_count * accumulate_grad_batches)

        return effective_batches * self.trainer.max_epochs

    # def on_before_optimizer_step(self, optimizer):
    #     # Compute the 2-norm for each layer
    #     # If using mixed precision, the gradients are already unscaled here
    #     norms = grad_norm(self.codec.parameters(), norm_type=2)
    #     self.log_dict(norms)

    def configure_optimizers(self):
        g_params = [p for p in self.codec.parameters() if p.requires_grad]
        opt_g = torch.optim.AdamW(
            g_params,
            lr=float(self.cfg.train.lr),
            betas=tuple(self.cfg.train.betas),
            weight_decay=float(self.cfg.train.weight_decay),
        )
        if not self.use_gan:
            return opt_g

        opt_d = torch.optim.AdamW(
            self.disc.parameters(),
            lr=float(self.cfg.train.lr_disc),
            betas=tuple(self.cfg.train.betas_disc),
            weight_decay=float(self.cfg.train.weight_decay),
        )
        return opt_g, opt_d

    def _get_feat(self, batch: Dict[str, Any]) -> torch.Tensor:
        if "feat" in batch:
            feat = batch["feat"].to(self.device)
            if feat.ndim == 2:
                feat = feat.unsqueeze(0)
            return feat
        wav_enc = batch["wav_enc"].to(self.device)  # (B, T)
        return self.feat_extractor(wav_enc)

    def _forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        feat = self._get_feat(batch)
        if "d_vector" in batch:
            dvec = batch["d_vector"].to(self.device)
            wav = forward_decoder(self.codec, feat=feat, d_vector=dvec)
        else:
            ref = batch["ref_wav"].to(self.device)
            wav = forward_decoder(self.codec, feat=feat, ref_wav_16k=ref)
        return wav

    def _align(self, y_hat: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # y_hat: (B,1,T), y: (B,T)
        y_hat = y_hat.squeeze(1)
        min_len = min(y_hat.shape[-1], y.shape[-1])
        return y_hat[..., :min_len], y[..., :min_len]

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        y = batch["wav_target"].to(self.device)

        if not self.use_gan:
            y_hat = self._forward(batch)
            y_hat, y = self._align(y_hat, y)
            loss = 0.0
            if self.cfg.loss.wave_l1_weight > 0:
                wave_l1_loss = float(self.cfg.loss.wave_l1_weight) * F.l1_loss(y_hat, y)
                loss = loss + wave_l1_loss
            if self.cfg.loss.mel_weight > 0:
                mel_loss = float(self.cfg.loss.mel_weight) * self.mel_loss(y_hat, y)
                loss = loss + mel_loss
            if self.cfg.loss.stft_weight > 0:
                stft_loss = float(self.cfg.loss.stft_weight) * self.stft_loss(y_hat, y)
                loss = loss + stft_loss

            self.log("train/loss", loss, prog_bar=True)
            self.log("train/wave_l1_loss", wave_l1_loss, prog_bar=False)
            self.log("train/mel_loss", mel_loss, prog_bar=False)
            self.log("train/stft_loss", stft_loss, prog_bar=False)

            return loss

        opt_g, opt_d = self.optimizers()

        # 1) Discriminator update
        with torch.no_grad():
            y_hat = self._forward(batch)
        y_hat_a, y_a = self._align(y_hat, y)
        y_a = y_a.unsqueeze(1)
        y_hat_a = y_hat_a.unsqueeze(1)

        d_real, _ = self.disc(y_a)
        d_fake, _ = self.disc(y_hat_a.detach())
        loss_d = discriminator_loss(d_real, d_fake)

        opt_d.zero_grad(set_to_none=True)
        self.manual_backward(loss_d)
        opt_d.step()

        # 2) Generator update
        y_hat = self._forward(batch)
        y_hat_a, y_a = self._align(y_hat, y)
        loss_g = 0.0
        if self.cfg.loss.wave_l1_weight > 0:
            wave_l1_loss = float(self.cfg.loss.wave_l1_weight) * F.l1_loss(y_hat_a, y_a)
            loss_g = loss_g + wave_l1_loss
        if self.cfg.loss.mel_weight > 0:
            mel_loss = float(self.cfg.loss.mel_weight) * self.mel_loss(y_hat_a, y_a)
            loss_g = loss_g + mel_loss
        if self.cfg.loss.stft_weight > 0:
            stft_loss = float(self.cfg.loss.stft_weight) * self.stft_loss(y_hat_a, y_a)
            loss_g = loss_g + stft_loss

        y_a = y_a.unsqueeze(1)
        y_hat_u = y_hat_a.unsqueeze(1)
        d_fake, fmap_fake = self.disc(y_hat_u)
        d_real, fmap_real = self.disc(y_a)

        adv = generator_adversarial_loss(d_fake)
        fm = feature_matching_loss(fmap_real, fmap_fake)
        loss_g = loss_g + float(self.cfg.loss.gan.adv_weight) * adv + float(self.cfg.loss.gan.fm_weight) * fm

        opt_g.zero_grad(set_to_none=True)
        self.manual_backward(loss_g)
        opt_g.step()

        self.log("train/loss_g", loss_g, prog_bar=True)
        self.log("train/wave_l1_loss", wave_l1_loss, prog_bar=False)
        self.log("train/mel_loss", mel_loss, prog_bar=False)
        self.log("train/stft_loss", stft_loss, prog_bar=False)
        self.log("train/loss_d", loss_d, prog_bar=False)

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        y = batch["wav_target"].to(self.device)
        y_hat = self._forward(batch)
        y_hat, y = self._align(y_hat, y)
        loss = 0.0
        if self.cfg.loss.wave_l1_weight > 0:
            wave_l1_loss = float(self.cfg.loss.wave_l1_weight) * F.l1_loss(y_hat, y)
            loss = loss + wave_l1_loss
        if self.cfg.loss.mel_weight > 0:
            mel_loss = float(self.cfg.loss.mel_weight) * self.mel_loss(y_hat, y)
            loss = loss + mel_loss
        if self.cfg.loss.stft_weight > 0:
            stft_loss = float(self.cfg.loss.stft_weight) * self.stft_loss(y_hat, y)
            loss = loss + stft_loss

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/wave_l1_loss", wave_l1_loss, prog_bar=False)
        self.log("val/mel_loss", mel_loss, prog_bar=False)
        self.log("val/stft_loss", stft_loss, prog_bar=False)

        # save one sample occasionally
        if batch_idx == 0:
            wav = y_hat[0].float().detach().cpu().numpy()
            generated_audios = {}
            generated_audios[f"val/sample_step{int(self.global_step)}"] = wandb.Audio(
                wav,
                sample_rate=self.target_sr,
            )
            wandb.log(generated_audios, step=int(self.global_step))


def build_dataset(cfg):
    train_items = load_jsonl(cfg.data.train_manifest)
    val_items = load_jsonl(cfg.data.val_manifest)

    seg = SegmentConfig(
        fps=int(cfg.data.fps),
        segment_seconds=float(cfg.data.segment_seconds),
        ref_seconds=float(cfg.data.ref_seconds),
        align_to_frames=bool(cfg.data.align_to_frames),
    )

    train_ds = BiCodecDecoderDataset(
        items=train_items,
        target_sample_rate=int(cfg.data.target_sample_rate),
        encoder_sample_rate=int(cfg.data.encoder_sample_rate),
        segment=seg,
        cache_dir=str(cfg.data.cache.dir) if cfg.data.cache.enabled else None,
        use_cached_dvector=bool(cfg.data.cache.cache_dvector),
    )
    val_ds = BiCodecDecoderDataset(
        items=val_items,
        target_sample_rate=int(cfg.data.target_sample_rate),
        encoder_sample_rate=int(cfg.data.encoder_sample_rate),
        segment=seg,
        cache_dir=str(cfg.data.val_cache.dir) if cfg.data.val_cache.enabled else None,
        use_cached_dvector=bool(cfg.data.val_cache.cache_dvector),
    )

    return train_ds, val_ds


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides, e.g. train.batch_size=8")
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)

    Path(cfg.exp.out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(cfg.exp.out_dir) / "resolved_config.yaml", "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg))

    L.seed_everything(int(cfg.exp.seed), workers=True)

    train_loader, val_loader = build_loaders(cfg)

    module = DecoderTrainModule(cfg)

    ckpt = ModelCheckpoint(
        dirpath=str(Path(cfg.exp.out_dir) / "checkpoints"),
        save_top_k=3,
        monitor="val/loss",
        mode="min",
        every_n_train_steps=int(cfg.train.save_every_n_steps),
        save_last=True,
    )
    lrmon = LearningRateMonitor(logging_interval="step")
    logger = TensorBoardLogger(save_dir=str(Path(cfg.exp.out_dir) / "tb"), name="")

    trainer = L.Trainer(
        default_root_dir=str(cfg.exp.out_dir),
        accelerator="gpu" if torch.cuda.is_available() and cfg.model.device == "cuda" else "cpu",
        devices=int(cfg.train.devices),
        precision=str(cfg.train.precision),
        max_steps=int(cfg.train.max_steps),
        val_check_interval=int(cfg.train.val_every_n_steps),
        log_every_n_steps=int(cfg.train.log_every_n_steps),
        callbacks=[ckpt, lrmon],
        logger=logger,
        gradient_clip_val=float(cfg.train.clip_grad),
        accumulate_grad_batches=int(cfg.train.grad_accum),
        enable_progress_bar=True,
    )

    trainer.fit(module, train_loader, val_loader)


if __name__ == "__main__":
    main()
