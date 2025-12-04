"""Training script for SVTRv2"""

import os
from datetime import datetime
from pathlib import Path

import lightning as L
import torch
import wandb
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from src.data.datamodule import SVTRv2DataModule
from src.lightning.lit_svtrv2 import LitSVTRv2
from src.models.svtrv2_model import SVTRv2Model
from yaml import safe_load


def load_config(config_path):
    """Load and process config"""
    with open(config_path, "r") as f:
        config = safe_load(f)
    return config


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=None,
        help="Number of GPUs per node (default: None for auto-detect)",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="[Deprecated] Use --devices instead. Number of GPUs per node",
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=1,
        help="Number of nodes (default: 1)",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=None,
        help="[Deprecated] Use --num_nodes instead. Number of nodes",
    )
    args = parser.parse_args()

    # Handle deprecated arguments and set defaults
    # devices: --devices 우선, 없으면 --gpus 사용
    if args.devices is None and args.gpus is not None:
        args.devices = args.gpus

    # num_nodes: --num_nodes 우선, 없으면 --nodes 사용, 둘 다 없으면 1
    if args.num_nodes == 1 and args.nodes is not None:
        args.num_nodes = args.nodes

    # Load config
    config = load_config(args.config)
    global_config = config.get("Global", {})
    arch_config = config.get("Architecture", {})

    # # Resolve character_dict_path relative to config file location
    # # This ensures the path works regardless of the current working directory
    # character_dict_path = global_config.get("character_dict_path")
    # if character_dict_path and not Path(character_dict_path).is_absolute():
    #     # Config file is in svtrv2/configs/, so go up one level to svtrv2/
    #     config_dir = Path(args.config).parent.parent
    #     character_dict_path = str(config_dir / character_dict_path.lstrip("./"))
    #     global_config["character_dict_path"] = character_dict_path
    #     # Also update in config for datamodule
    #     config["Global"]["character_dict_path"] = character_dict_path

    # wandb 설정
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "local")
    run_name = f"job{slurm_job_id}_{datetime.now().strftime('%y%m%d_%H%M')}"
    wandb_logger = WandbLogger(
        project="LPR-SVTRv2",
        name=run_name,
    )

    # Build data module from config
    datamodule = SVTRv2DataModule.from_config(config)

    # Calculate character dictionary size
    character_dict_path = global_config.get("character_dict_path")
    use_space_char = global_config.get("use_space_char", False)

    # Load character dictionary to get size
    from src.data.label_encode import CTCLabelEncoder

    label_encoder = CTCLabelEncoder(
        character_dict_path=character_dict_path,
        use_space_char=use_space_char,
        max_text_length=global_config.get("max_text_length", 25),
    )
    num_classes = len(label_encoder.character)  # Includes blank token

    # Set decoder out_channels to match character dictionary size
    if "Decoder" in arch_config:
        arch_config["Decoder"]["out_channels"] = num_classes
    else:
        arch_config["Decoder"] = {"out_channels": num_classes}

    # Build model
    model = SVTRv2Model(arch_config)

    # Build Lightning module
    lit_module = LitSVTRv2(model, config)

    # Build trainer
    callbacks = []

    # Learning Rate Monitor 설정
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)

    # Model checkpoint callback
    output_dir = Path(global_config.get("output_dir", "./output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename="checkpoint-epoch{epoch:02d}-{val_loss:.3f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    callbacks.append(checkpoint_callback)

    trainer = L.Trainer(
        max_epochs=global_config.get("epoch_num", 20),
        accelerator="gpu",
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy="ddp",
        precision=16 if global_config.get("use_amp", False) else 32,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=global_config.get("print_batch_step", 10),
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # Train
    trainer.fit(lit_module, datamodule=datamodule, ckpt_path=args.resume)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    wandb.login(key="53f960c86b81377b89feb5d30c90ddc6c3810d3a")
    main()
