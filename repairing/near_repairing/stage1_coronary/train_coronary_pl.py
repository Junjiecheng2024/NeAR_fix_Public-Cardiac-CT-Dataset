"""PyTorch Lightning trainer for Coronary (stage1) NeAR.

Usage:
  - For local single-GPU debugging:
      python train_coronary_pl.py --devices 1
  - For multi-GPU (on cluster with 4 A100):
      python train_coronary_pl.py --devices 4 --strategy ddp

This script wraps the existing dataset + GatherGridsFromVolumes logic into a
LightningDataModule and uses the provided `CoronaryNeARLightningModule`.
It integrates WandB logging and ModelCheckpoint callbacks.
"""

import os
import sys
import time
import argparse

# Ensure project root is on sys.path (same behavior as original scripts)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from lightning_module import CoronaryNeARLightningModule
from near.datasets.cardiac_dataset import CardiacClassDatasetWithBiasedSampling
from near.models.nn3d.grid import GatherGridsFromVolumes


class GridGatherWrapperDataset:
    """Wraps original dataset to return (indices, grids, labels) so DataLoader
    can be used directly with LightningModule which expects these tensors.
    """
    def __init__(self, base_dataset, gather_fn):
        self.base = base_dataset
        self.gather_fn = gather_fn

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        # base returns (indices, shape) similar to legacy script
        item = self.base[idx]
        # item could be (index,) or (index, shape) depending on dataset
        if isinstance(item, tuple) and len(item) >= 2:
            indices, shape = item[0], item[1]
        else:
            indices = item
            shape = None

        # gather_fn returns (coords?, grids, labels) in the legacy code
        # we ignore the first returned element and keep grids and labels
        _, grids, labels = self.gather_fn(shape)

        return indices, grids, labels


class CoronaryDataModule:
    def __init__(self, cfg):
        self.cfg = cfg

    def setup(self, stage=None):
        data_path = self.cfg['data_path']

        # Train and eval datasets (we intentionally use same dataset to overfit)
        train_ds = CardiacClassDatasetWithBiasedSampling(
            root=data_path,
            class_name=self.cfg['class_name'],
            resolution=self.cfg['target_resolution'],
            n_samples=self.cfg['n_training_samples'],
            sampling_bias_ratio=self.cfg['sampling_bias_ratio'],
            sampling_dilation_radius=self.cfg['sampling_dilation_radius'],
            class_index=self.cfg['class_index']
        )

        eval_ds = CardiacClassDatasetWithBiasedSampling(
            root=data_path,
            class_name=self.cfg['class_name'],
            resolution=self.cfg['target_resolution'],
            n_samples=self.cfg['n_training_samples'],
            sampling_bias_ratio=self.cfg['sampling_bias_ratio'],
            sampling_dilation_radius=self.cfg['sampling_dilation_radius'],
            class_index=self.cfg['class_index']
        )

        # Gather functions
        self.train_gather = GatherGridsFromVolumes(
            self.cfg['training_resolution'],
            grid_noise=self.cfg.get('grid_noise', None),
            uniform_grid_noise=self.cfg.get('uniform_grid_noise', True),
            boundary_bias_ratio=self.cfg.get('sampling_bias_ratio', 0.5),
            boundary_dilation_radius=self.cfg.get('sampling_dilation_radius', 2)
        )

        self.eval_gather = GatherGridsFromVolumes(
            self.cfg['target_resolution'],
            grid_noise=None,
            boundary_bias_ratio=0.0
        )

        # Wrap datasets so they directly yield (indices, grids, labels)
        self.train_dataset = GridGatherWrapperDataset(train_ds, self.train_gather)
        self.eval_dataset = GridGatherWrapperDataset(eval_ds, self.eval_gather)

    def train_dataloader(self):
        from torch.utils.data import DataLoader
        return DataLoader(self.train_dataset,
                          batch_size=self.cfg['batch_size'],
                          shuffle=True,
                          pin_memory=torch.cuda.is_available(),
                          num_workers=self.cfg['n_workers'])

    def val_dataloader(self):
        from torch.utils.data import DataLoader
        return DataLoader(self.eval_dataset,
                          batch_size=self.cfg['eval_batch_size'],
                          shuffle=False,
                          pin_memory=torch.cuda.is_available(),
                          num_workers=self.cfg['n_workers'])


def main(args):
    # Load config
    import importlib
    config_name = args.config if hasattr(args, 'config') and args.config else 'config_coronary'
    cfg_module = importlib.import_module(config_name)
    cfg = cfg_module.cfg
    print(f"\nLoaded config from: {config_name}")
    print(f"Training for {cfg['n_epochs']} epochs\n")
    
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
    n_samples = len(dm.train_dataset.base)
    pl_module = CoronaryNeARLightningModule(
        n_samples=n_samples,
        latent_dimension=cfg.get('latent_dimension', 256),
        decoder_channels=cfg.get('decoder_channels', [64, 48, 32, 16]),
        lr=cfg.get('lr', 1e-3),
        l2_penalty_weight=cfg.get('l2_penalty_weight', 1e-4),
        use_cosine_schedule=cfg.get('use_cosine_schedule', False),
        warmup_ratio=cfg.get('warmup_ratio', 0.01),
        total_steps=None,  # optional, not required for MultiStepLR
        milestones=cfg.get('milestones', [100, 200]),
        gamma=cfg.get('gamma', 0.5),
    )

    # Attempt to load existing checkpoint weights (if any)
    resume_ckpt = cfg.get('resume_checkpoint', None)
    if resume_ckpt and os.path.exists(resume_ckpt):
        try:
            state = torch.load(resume_ckpt, map_location='cpu')
            # Try loading into underlying model if keys match
            try:
                pl_module.model.load_state_dict(state)
                print(f"Loaded pretrained weights into pl_module.model from {resume_ckpt}")
            except Exception:
                # Fallback: try loading into module directly
                pl_module.load_state_dict(state, strict=False)
                print(f"Loaded checkpoint into pl_module (partial) from {resume_ckpt}")
        except Exception as e:
            print(f"Warning: failed to load resume checkpoint: {e}")

    # WandB logger
    wandb_logger = WandbLogger(project='NeAR_stage1_coronary', name=cfg['run_flag'])

    # Callbacks
    ckpt_cb = ModelCheckpoint(dirpath=base_path,
                              filename='best',
                              monitor='val/shape_loss',
                              mode='min',
                              save_top_k=1,
                              save_last=True)

    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Trainer
    precision = 16 if cfg.get('use_amp', False) else 32

    trainer = Trainer(
        logger=wandb_logger,
        callbacks=[ckpt_cb, lr_monitor],
        max_epochs=cfg['n_epochs'],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=args.devices,
        strategy=args.strategy if args.strategy else None,
        precision=precision,
        accumulate_grad_batches=cfg.get('gradient_accumulation_steps', 1),
        check_val_every_n_epoch=cfg.get('eval_interval', 1),
        log_every_n_steps=50,
        deterministic=True,
    )

    # Fit
    trainer.fit(pl_module, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--devices', type=int, default=1, help='Number of devices (GPUs) to use')
    parser.add_argument('--strategy', type=str, default=None, help='DDP strategy (e.g., ddp)')
    args = parser.parse_args()

    main(args)