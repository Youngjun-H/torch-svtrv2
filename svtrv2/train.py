"""Training script for SVTRv2"""
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
        "--gpus",
        type=int,
        default=None,
        help="Number of GPUs per node (default: 1, None for auto-detect)",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=1,
        help="Number of nodes (default: 1)",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    global_config = config.get("Global", {})
    arch_config = config.get("Architecture", {})

    # wandb 설정
    wandb_logger = WandbLogger(project="LPR-SVTRv2") 

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
        filename="checkpoint-epoch{epoch:02d}-valloss{val_loss:.3f}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False,
    )
    callbacks.append(checkpoint_callback)

    trainer = L.Trainer(
        max_epochs=global_config.get("epoch_num", 20),
        accelerator="gpu",
        devices=args.gpus,
        num_nodes=args.nodes,
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
