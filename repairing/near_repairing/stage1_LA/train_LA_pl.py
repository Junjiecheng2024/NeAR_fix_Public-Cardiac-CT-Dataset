"""PyTorch Lightning trainer for LA (stage1) NeAR.
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

from lightning_module import LANeARLightningModule
from near.datasets.cardiac_dataset import CardiacClassDatasetWithBiasedSampling
from near.models.nn3d.grid import GatherGridsFromVolumes


class LADataModule:
    def __init__(self, cfg):
        self.cfg = cfg
        self.train_gather_fn = None
        self.eval_gather_fn = None

    def setup(self, stage=None):
        self.train_dataset = CardiacClassDatasetWithBiasedSampling(
            self.cfg['data_path'],
            class_name=self.cfg['class_name'],
            class_index=self.cfg['class_index']
        )

        self.eval_dataset = CardiacClassDatasetWithBiasedSampling(
            self.cfg['data_path'],
            class_name=self.cfg['class_name'],
            class_index=self.cfg['class_index']
        )

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
            boundary_bias_ratio=0.0,
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
    config_name = args.config if hasattr(args, 'config') and args.config else 'config_LA'
    cfg_module = importlib.import_module(config_name)
    cfg = cfg_module.cfg
    
    print(f"\n{'='*70}")
    print(f"Loading config from: {config_name}")
    print(f"Training epochs: {cfg['n_epochs']}")
    print(f"{'='*70}\n")

    cfg['run_flag'] += time.strftime("%y%m%d_%H%M%S")
    base_path = os.path.join(cfg['base_path'], cfg['run_flag'])
    os.makedirs(base_path, exist_ok=True)

    dm = LADataModule(cfg)
    dm.setup()

    n_samples = len(dm.train_dataset)
    pl_module = LANeARLightningModule(
        n_samples=n_samples,
        train_gather_fn=dm.train_gather_fn,
        eval_gather_fn=dm.eval_gather_fn,
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

    resume_ckpt = cfg.get('resume_checkpoint', None)
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

    wandb_logger = WandbLogger(project='NeAR_stage1_LA', name=cfg['run_flag'])

    ckpt_cb = ModelCheckpoint(
        dirpath=base_path,
        filename='best',
        monitor='val/shape_loss',
        mode='min',
        save_top_k=1,
        save_last=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

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

    trainer.fit(pl_module, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())

    print(f"\nTraining completed! Best checkpoint saved to: {base_path}/best.ckpt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--devices', type=int, default=1, help='Number of devices (GPUs) to use')
    parser.add_argument('--strategy', type=str, default=None, help='DDP strategy (e.g., ddp)')
    parser.add_argument('--config', type=str, default='config_LA', help='Config module name (without .py)')
    args = parser.parse_args()

    main(args)
