#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch Lightning trainer for NeAR v2.0 Tier2 (Shape + Appearance).
For training Coronary and other small structures with class-specific crops.
"""
import os
import sys
import argparse
import time
import importlib.util

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

# Handle distributed training: detect rank from SLURM or torchrun
local_rank = int(os.environ.get('SLURM_LOCALID', os.environ.get('LOCAL_RANK', 0)))
global_rank = int(os.environ.get('SLURM_PROCID', os.environ.get('RANK', 0)))
world_size = int(os.environ.get('SLURM_NTASKS', os.environ.get('WORLD_SIZE', 1)))

if world_size > 1:
    print(f"[Rank {global_rank}/{world_size}, LocalRank {local_rank}] Distributed training detected")

# WandB login for non-interactive sbatch jobs (only on rank 0)
import wandb
if global_rank == 0:
    wandb.login(key="d6891a1bb4397a24519ef1b36091aa1b77ea67e1")

from lightning_module import CoronaryTier2LightningModule
from near.datasets.coronary_tier2_dataset import CoronaryTier2Dataset
from near.models.nn3d.grid import GatherGridsFromVolumes


class Tier2DataModule:
    """Data module for Tier2 training with Shape + Appearance."""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.train_gather_fn = None
        self.eval_gather_fn = None
        self.train_dataset = None
        self.eval_dataset = None
    
    def setup(self, stage=None):
        # Get resolution from first sample if not specified
        if self.cfg.get('target_resolution') is None:
            # Load one sample to get shape
            temp_dataset = CoronaryTier2Dataset(
                root=self.cfg['data_path'],
                resolution=None,
                n_samples=1,
                use_appearance=False
            )
            sample = temp_dataset[0]
            sample_shape = sample['shape'].shape[-1]  # Assuming cubic
            self.cfg['_inferred_resolution'] = sample_shape
            grid_resolution = sample_shape
        else:
            grid_resolution = self.cfg['target_resolution']
            self.cfg['_inferred_resolution'] = grid_resolution
        
        print(f"[DataModule] Using grid resolution: {grid_resolution}")
        
        # Create datasets
        self.train_dataset = CoronaryTier2Dataset(
            root=self.cfg['data_path'],
            resolution=self.cfg.get('target_resolution'),
            n_samples=self.cfg.get('n_training_samples'),
            use_appearance=self.cfg.get('use_appearance', True),
            boundary_bias_ratio=self.cfg.get('sampling_bias_ratio', 0.5),
            boundary_dilation_radius=self.cfg.get('sampling_dilation_radius', 3),
            augment=self.cfg.get('augment', True)
        )
        
        self.eval_dataset = CoronaryTier2Dataset(
            root=self.cfg['data_path'],
            resolution=self.cfg.get('target_resolution'),
            n_samples=self.cfg.get('n_training_samples'),
            use_appearance=self.cfg.get('use_appearance', True),
            boundary_bias_ratio=0.0,  # No bias for eval
            boundary_dilation_radius=self.cfg.get('sampling_dilation_radius', 3),
            augment=False
        )
        
        # Grid gather functions
        self.train_gather_fn = GatherGridsFromVolumes(
            resolution=grid_resolution,
            grid_noise=self.cfg.get('grid_noise', 0),
            uniform_grid_noise=self.cfg.get('uniform_grid_noise', True),
            label_interpolation_mode='nearest',
            boundary_bias_ratio=self.cfg.get('sampling_bias_ratio', 0.5),
            boundary_dilation_radius=self.cfg.get('sampling_dilation_radius', 3)
        )
        
        self.eval_gather_fn = GatherGridsFromVolumes(
            resolution=grid_resolution,
            grid_noise=0,
            uniform_grid_noise=True,
            label_interpolation_mode='nearest',
            boundary_bias_ratio=0.0,
            boundary_dilation_radius=3
        )
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.cfg.get('batch_size', 1),
            shuffle=True,
            num_workers=self.cfg.get('n_workers', 4),
            pin_memory=True,
            drop_last=True
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.eval_dataset,
            batch_size=self.cfg.get('eval_batch_size', 1),
            shuffle=False,
            num_workers=self.cfg.get('n_workers', 4),
            pin_memory=True
        )


def load_config(config_path: str, class_name: str = None):
    """
    Load config from Python file.
    
    Args:
        config_path: Path to config.py file
        class_name: Optional class name to load specific config (e.g., 'coronary', 'aorta')
    
    Returns:
        Configuration dictionary
    """
    spec = importlib.util.spec_from_file_location("config", config_path)
    cfg_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_module)
    
    if class_name is not None:
        # Use get_config function from config module to get class-specific config
        if hasattr(cfg_module, 'get_config'):
            config_obj = cfg_module.get_config(class_name)
            return config_obj.to_dict()
        else:
            raise ValueError(f"Config module does not have get_config function")
    else:
        # Backward compatibility: use default cfg
        return cfg_module.cfg


def main(args):
    cfg = load_config(args.config, args.class_name)
    
    print(f"\n{'='*70}")
    print("NeAR v2.0 Tier2 Training (Shape + Appearance)")
    print(f"{'='*70}")
    print(f"Config: {args.config}")
    print(f"Class: {cfg['class_name']}")
    print(f"Epochs: {cfg['n_epochs']}")
    print(f"Data Path: {cfg['data_path']}")
    print(f"Use Appearance: {cfg.get('use_appearance', True)}")
    print(f"Use Context: {cfg.get('use_context', True)}")
    print(f"{'='*70}\n")
    
    # Add timestamp to run flag
    cfg['run_flag'] += time.strftime("%y%m%d_%H%M%S")
    base_path = os.path.join(cfg['base_path'], cfg['run_flag'])
    os.makedirs(base_path, exist_ok=True)
    
    # Setup data
    dm = Tier2DataModule(cfg)
    dm.setup()
    
    n_samples = len(dm.train_dataset)
    print(f"Number of training samples: {n_samples}")
    
    # Create model
    pl_module = CoronaryTier2LightningModule(
        n_samples=n_samples,
        grid_gather_fn=dm.train_gather_fn,
        latent_dimension=cfg.get('latent_dimension', 256),
        decoder_channels=cfg.get('decoder_channels', [64, 48, 32, 16]),
        appearance_channels=cfg.get('appearance_channels', 64),
        use_context=cfg.get('use_context', True),
        lr=cfg.get('lr', 5e-4),
        l2_penalty_weight=cfg.get('l2_penalty_weight', 1e-4),
        dice_weight=cfg.get('dice_weight', 0.3),
        tversky_weight=cfg.get('tversky_weight', 0.35),
        boundary_dice_weight=cfg.get('boundary_dice_weight', 0.2),
        focal_weight=cfg.get('focal_weight', 0.1),
        topk_weight=cfg.get('topk_weight', 0.05),
        use_cosine_schedule=cfg.get('use_cosine_schedule', True),
        warmup_ratio=cfg.get('warmup_ratio', 0.02),
    )
    
    # Resume from checkpoint if specified
    resume_ckpt = cfg.get('resume_checkpoint', None)
    if resume_ckpt and os.path.exists(resume_ckpt):
        print(f"Loading checkpoint: {resume_ckpt}")
        try:
            state = torch.load(resume_ckpt, map_location='cpu')
            if 'state_dict' in state:
                pl_module.load_state_dict(state['state_dict'], strict=False)
            else:
                pl_module.model.load_state_dict(state, strict=False)
            print("Checkpoint loaded successfully!")
        except Exception as e:
            print(f"Warning: Failed to load checkpoint: {e}")
    
    # Setup logger
    wandb_logger = WandbLogger(
        project=f"NeAR_v2_Tier2_{cfg['class_name']}", 
        name=cfg['run_flag']
    )
    
    # Callbacks
    ckpt_cb = ModelCheckpoint(
        dirpath=base_path,
        filename='best',
        monitor='val/dice_score',
        mode='max',
        save_top_k=1,
        save_last=True
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Trainer
    precision = '16-mixed' if cfg.get('use_amp', True) else 32
    
    # Use Lightning's native multi-GPU support
    # When devices > 1, Lightning will spawn worker processes and handle DDP
    num_devices = args.devices if args.devices != 'auto' else 'auto'
    use_ddp = (isinstance(num_devices, int) and num_devices > 1) or num_devices == 'auto'
    
    trainer = Trainer(
        logger=wandb_logger,
        callbacks=[ckpt_cb, lr_monitor],
        max_epochs=cfg['n_epochs'],
        accelerator='gpu',
        devices=num_devices,
        num_nodes=1,
        strategy='ddp' if use_ddp else 'auto',
        precision=precision,
        accumulate_grad_batches=cfg.get('gradient_accumulation_steps', 1),
        check_val_every_n_epoch=cfg.get('eval_interval', 5),
        log_every_n_steps=50,
        deterministic=False,
        use_distributed_sampler=use_ddp,
    )
    
    # Train
    trainer.fit(
        pl_module, 
        train_dataloaders=dm.train_dataloader(), 
        val_dataloaders=dm.val_dataloader()
    )
    
    print(f"\nTraining completed! Best checkpoint saved to: {base_path}/best.ckpt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NeAR v2.0 Tier2 Training")
    parser.add_argument('--devices', type=str, default='1', 
                        help='Number of GPUs or "auto"')
    parser.add_argument('--strategy', type=str, default=None, 
                        help='DDP strategy (e.g., ddp)')
    parser.add_argument('--config', type=str, required=True, 
                        help='Path to config file')
    parser.add_argument('--class_name', type=str, default=None,
                        help='Class name to train (e.g., coronary, aorta, la, lv, ra, rv, pa, pv, laa, myocardium)')
    args = parser.parse_args()
    
    # Parse devices: can be int or "auto"
    if args.devices == 'auto':
        args.devices = 'auto'
    else:
        args.devices = int(args.devices)
    
    main(args)

