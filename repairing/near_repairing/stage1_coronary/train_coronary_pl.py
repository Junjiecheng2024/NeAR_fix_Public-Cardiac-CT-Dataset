"""PyTorch Lightning trainer for Coronary (stage1) NeAR.

Usage:
  - For local single-GPU debugging:
      python train_coronary_pl.py --devices 1 --config config_coronary_debug
  - For multi-GPU (on cluster with 4 A100):
      python train_coronary_pl.py --devices 4 --strategy ddp --config config_coronary

This script follows the original train_coronary.py design:
- DataLoader returns (indices, shape) tuples
- gather_fn is called in training_step, not in Dataset.__getitem__
- This avoids collate/pin_memory issues
"""
import os
import sys
import argparse
import time
import importlib

# 添加 near 模块路径到 Python 路径
near_root = '/projappl/project_2016517/chengjun/NeAR_fix_Public-Cardiac-CT-Dataset'
if near_root not in sys.path:
    sys.path.insert(0, near_root)

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from lightning_module import CoronaryNeARLightningModule
from near.datasets.cardiac_dataset import CardiacClassDatasetWithBiasedSampling
from near.models.nn3d.grid import GatherGridsFromVolumes


class CoronaryDataModule:
    """
    DataModule for Coronary training.
    
    Key design:
    - Datasets return (indices, shape) - simple tuples
    - gather_fn is stored and passed to LightningModule
    - gather_fn is called in training_step, NOT in Dataset
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.train_gather_fn = None
        self.eval_gather_fn = None

    def setup(self, stage=None):
        # Training dataset - 最简化版本，只传必需参数
        self.train_dataset = CardiacClassDatasetWithBiasedSampling(
            self.cfg['data_path'],
            class_name=self.cfg['class_name'],
            class_index=self.cfg['class_index']
        )

        # Validation dataset - 使用相同的数据集（目标是过拟合）
        self.eval_dataset = CardiacClassDatasetWithBiasedSampling(
            self.cfg['data_path'],
            class_name=self.cfg['class_name'],
            class_index=self.cfg['class_index']
        )

        # Create gather functions (to be used in LightningModule)
        # 所有的采样参数都在这里配置
        self.train_gather_fn = GatherGridsFromVolumes(
            resolution=self.cfg['training_resolution'],
            grid_noise=self.cfg.get('grid_noise', 0),
            uniform_grid_noise=self.cfg.get('uniform_grid_noise', True),
            label_interpolation_mode='nearest',
            boundary_bias_ratio=self.cfg.get('sampling_bias_ratio', 0.0),
            boundary_dilation_radius=self.cfg.get('sampling_dilation_radius', 2)
        )

        self.eval_gather_fn = GatherGridsFromVolumes(
            resolution=self.cfg['target_resolution'],
            grid_noise=0,
            uniform_grid_noise=True,
            label_interpolation_mode='nearest',
            boundary_bias_ratio=0.0,  # No bias for evaluation
            boundary_dilation_radius=2
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.cfg['batch_size'],
            shuffle=True,
            num_workers=self.cfg.get('n_workers', 0),
            pin_memory=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.eval_dataset,
            batch_size=self.cfg.get('eval_batch_size', self.cfg['batch_size']),
            shuffle=False,
            num_workers=self.cfg.get('n_workers', 0),
            pin_memory=True
        )


def main(args):
    # Load config
    config_name = args.config if hasattr(args, 'config') and args.config else 'config_coronary'
    cfg_module = importlib.import_module(config_name)
    cfg = cfg_module.cfg
    
    print(f"\n{'='*70}")
    print(f"Loading config from: {config_name}")
    print(f"Training epochs: {cfg['n_epochs']}")
    print(f"{'='*70}\n")

    # Create base checkpoint path
    cfg['run_flag'] += time.strftime("%y%m%d_%H%M%S")
    base_path = os.path.join(cfg['base_path'], cfg['run_flag'])
    os.makedirs(base_path, exist_ok=True)

    # DataModule
    dm = CoronaryDataModule(cfg)
    dm.setup()

    # LightningModule
    n_samples = len(dm.train_dataset)
    pl_module = CoronaryNeARLightningModule(
        n_samples=n_samples,
        train_gather_fn=dm.train_gather_fn,  # 传入 gather_fn
        eval_gather_fn=dm.eval_gather_fn,    # 传入 gather_fn
        latent_dimension=cfg.get('latent_dimension', 256),
        decoder_channels=cfg.get('decoder_channels', [64, 48, 32, 16]),
        lr=cfg.get('lr', 1e-3),
        l2_penalty_weight=cfg.get('l2_penalty_weight', 1e-4),
        use_cosine_schedule=cfg.get('use_cosine_schedule', False),
        warmup_ratio=cfg.get('warmup_ratio', 0.01),
        total_steps=None,
        milestones=cfg.get('milestones', [100, 200]),
        gamma=cfg.get('gamma', 0.5),
    )

    # Attempt to load existing checkpoint weights (if any)
    resume_ckpt = cfg.get('resume_checkpoint', None)
    print(f"[DEBUG] resume_checkpoint in cfg: {resume_ckpt}")
    if resume_ckpt:
        print(f"[DEBUG] exists? {os.path.exists(resume_ckpt)}")

    if resume_ckpt and os.path.exists(resume_ckpt):
        try:
            state = torch.load(resume_ckpt, map_location='cpu')
            try:
                pl_module.model.load_state_dict(state)
                print(f"Loaded pretrained weights into pl_module.model from {resume_ckpt}")
            except Exception:
                pl_module.load_state_dict(state, strict=False)
                print(f"Loaded checkpoint into pl_module (partial) from {resume_ckpt}")
        except Exception as e:
            print(f"Warning: failed to load resume checkpoint: {e}")


    # WandB logger
    wandb_logger = WandbLogger(project='NeAR_stage1_coronary', name=cfg['run_flag'])

    # Callbacks
    ckpt_cb = ModelCheckpoint(
        dirpath=base_path,
        filename='best',
        monitor='val/shape_loss',
        mode='min',
        save_top_k=1,
        save_last=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Trainer
    precision = 16 if cfg.get('use_amp', False) else 32

    trainer = Trainer(
        logger=wandb_logger,
        callbacks=[ckpt_cb, lr_monitor],
        max_epochs=cfg['n_epochs'],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=args.devices,
        strategy=args.strategy if args.strategy else 'auto',
        precision=precision,
        accumulate_grad_batches=cfg.get('gradient_accumulation_steps', 1),
        check_val_every_n_epoch=cfg.get('eval_interval', 1),
        log_every_n_steps=50,
        deterministic=False,
    )

    # Fit
    trainer.fit(pl_module, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())

    print(f"\nTraining completed! Best checkpoint saved to: {base_path}/best.ckpt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--devices', type=int, default=1, help='Number of devices (GPUs) to use')
    parser.add_argument('--strategy', type=str, default=None, help='DDP strategy (e.g., ddp)')
    parser.add_argument('--config', type=str, default='config_coronary', help='Config module name (without .py)')
    args = parser.parse_args()

    main(args)