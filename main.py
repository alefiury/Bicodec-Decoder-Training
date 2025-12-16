import os
import argparse

import wandb
import torch
from omegaconf import OmegaConf
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from bicodec_train.train import DecoderTrainModule

torch.autograd.set_detect_anomaly(True) # for debugging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_path",
        required=True,
        type=str,
        help="YAML file with configurations"
    )
    parser.add_argument(
        "-g",
        "--gpu",
        default=0,
        required=False,
        type=int
    )
    parser.add_argument(
        "-ck",
        "--checkpoint-dir",
        required=False,
        type=str,
        default="./checkpoints/Spark-TTS-Decoder",
    )

    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)

    tags = []
    tags += config.tags  # add tags defined for experiments
    exp_title = config.title

    wandb.init(
        project=config.wandb_project_name,
        name=exp_title,
        tags=tags,
        entity=config.wandb_entity,
        config=OmegaConf.to_container(config, resolve=True)
    )

    logger = WandbLogger(
        project=config.wandb_project_name,
        name=exp_title,
        tags=tags,
        entity=config.wandb_entity,
        config=OmegaConf.to_container(config, resolve=True)
    )

    config["model_checkpoint"].pop("dirpath")

    callbacks = [
        ModelCheckpoint(**config["model_checkpoint"]),
        LearningRateMonitor("step"),
    ]

    model = DecoderTrainModule(config)

    # print(model)

    trainer = Trainer(
        **config["trainer"],
        logger=logger,
        callbacks=callbacks,
        devices=[args.gpu],
        default_root_dir=os.path.join(args.checkpoint_dir, config["title"]),
        enable_progress_bar=True,
    )

    trainer.fit(model)


if __name__ == "__main__":
    main()
