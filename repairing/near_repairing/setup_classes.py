import os

# Base directory
base_dir = "/home/user/persistent/NeAR_fix_Public-Cardiac-CT-Dataset/repairing/near_repairing"

# Class definitions: (Name, Index)
classes = [
    ("Myocardium", 1),
    ("LA", 2),
    ("LV", 3),
    ("RA", 4),
    ("Aorta", 6),
    ("PA", 7),
    ("LAA", 8),
    ("PV", 10)
]

# Templates
config_template = """\"\"\"
阶段一：{class_name} 训练配置文件

配置说明：
- 训练分辨率：128³
- 批处理：batch_size=1, gradient_accumulation=4
- 学习率：5e-4
- 边界偏采样：50%→30%→10%
- 损失函数：70% Dice + 30% Focal (gamma=4.0)
- 验证间隔：每10轮
- 混合精度：AMP开启

Configuration for {class_name} (class {class_index}) single-class refinement.
Phase 1: Shape-only NeAR training to refine noisy labels.
\"\"\"

cfg = dict()

# Paths
cfg["base_path"] = "./checkpoints"
cfg["run_flag"] = "{class_name}_class{class_index}_shape_only_"
cfg['data_path'] = '/scratch/project_2016517/junjie/dataset/near_format_data'

# Class information
cfg['class_name'] = '{class_name}'
cfg['class_index'] = {class_index}

# Training parameters
cfg["n_epochs"] = 400
# Note: milestones已废弃，现在使用Cosine Annealing scheduler

# Model parameters
cfg['appearance'] = False  # Shape-only mode
cfg['decoder_channels'] = [64, 48, 32, 16]
cfg['latent_dimension'] = 256

# Data parameters
cfg["training_resolution"] = 128  
cfg["target_resolution"] = 128   
cfg["n_training_samples"] = None 

# Optimization
cfg["lr"] = 1e-3
cfg["batch_size"] = 1  
cfg["gradient_accumulation_steps"] = 6
cfg["eval_batch_size"] = 1  
cfg["n_workers"] = 8

# Learning rate schedule 
cfg["use_cosine_schedule"] = True  
cfg["gamma"] = 0.5
cfg["warmup_ratio"] = 0.01

# Mixed precision training (AMP)
cfg["use_amp"] = True

# Validation interval
cfg["eval_interval"] = 5

# Sampling strategy
cfg["grid_noise"] = 0
cfg["uniform_grid_noise"] = True
cfg["sampling_bias_ratio"] = 0.05
cfg["sampling_dilation_radius"] = 2

# Loss weights
cfg['l2_penalty_weight'] = 1e-4

# Resume training from checkpoint
cfg["resume_checkpoint"] = None

# Note: We want overfitting to get the best refined labels
"""

lightning_module_template = """\"\"\"
PyTorch Lightning Module for {class_name} NeAR Training
\"\"\"
import os
import sys

# 添加 near 模块路径到 Python 路径
near_root = '/projappl/project_2016517/chengjun/NeAR_fix_Public-Cardiac-CT-Dataset'
if near_root not in sys.path:
    sys.path.insert(0, near_root)

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Any, Optional

from near.models.nn3d.model_shape_only import EmbeddingDecoderShapeOnly
from near.models.losses import latent_l2_penalty, dice_score


class FocalLoss(nn.Module):
    \"\"\"
    Focal Loss for addressing class imbalance.
    \"\"\"
    def __init__(self, alpha=0.25, gamma=4.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


class {class_name}NeARLightningModule(pl.LightningModule):
    \"\"\"
    PyTorch Lightning wrapper for NeAR {class_name} training.
    \"\"\"
    
    def __init__(
        self,
        n_samples: int,
        train_gather_fn,
        eval_gather_fn,
        latent_dimension: int = 256,
        decoder_channels: list = [64, 48, 32, 16],
        lr: float = 1e-3,
        l2_penalty_weight: float = 1e-4,
        use_cosine_schedule: bool = False,
        warmup_ratio: float = 0.01,
        total_steps: Optional[int] = None,
        milestones: list = [100, 200],
        gamma: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['train_gather_fn', 'eval_gather_fn'])
        
        self.train_gather_fn = train_gather_fn
        self.eval_gather_fn = eval_gather_fn
        
        self.model = EmbeddingDecoderShapeOnly(
            n_samples=n_samples,
            latent_dimension=latent_dimension,
            decoder_channels=decoder_channels,
        )
        
        self.focal_loss_fn = FocalLoss(alpha=0.25, gamma=4.0)
        
        self.lr = lr
        self.l2_penalty_weight = l2_penalty_weight
        self.use_cosine_schedule = use_cosine_schedule
        self.warmup_ratio = warmup_ratio
        self.total_steps = total_steps
        self.milestones = milestones
        self.gamma = gamma
    
    def forward(self, indices, grids):
        return self.model(indices, grids)
    
    def training_step(self, batch, batch_idx):
        indices, shape = batch
        _, grids, labels = self.train_gather_fn(shape)
        
        grids = grids.to(self.device)
        labels = labels.to(self.device)
        
        pred_logit_shape, encoded = self(indices, grids)
        
        bce_loss = F.binary_cross_entropy_with_logits(pred_logit_shape, labels)
        focal_loss = self.focal_loss_fn(pred_logit_shape, labels)
        
        pred_prob = torch.sigmoid(pred_logit_shape)
        dice = dice_score(pred_prob, labels)
        dice_loss = 1.0 - dice
        
        shape_loss = 0.85 * dice_loss + 0.15 * focal_loss
        l2_loss = latent_l2_penalty(encoded)
        
        total_loss = shape_loss + self.l2_penalty_weight * l2_loss
        
        self.log('train/total_loss', total_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train/shape_loss', shape_loss, on_step=True, on_epoch=True)
        self.log('train/dice_loss', dice_loss, on_step=True, on_epoch=True)
        self.log('train/focal_loss', focal_loss, on_step=True, on_epoch=True)
        self.log('train/bce_loss', bce_loss, on_step=True, on_epoch=True)
        self.log('train/l2_loss', l2_loss, on_step=True, on_epoch=True)
        self.log('train/dice_score', dice, prog_bar=True, on_step=True, on_epoch=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        indices, shape = batch
        _, grids, labels = self.eval_gather_fn(shape)
        
        grids = grids.to(self.device)
        labels = labels.to(self.device)
        
        pred_logit_shape, encoded = self(indices, grids)
        
        bce_loss = F.binary_cross_entropy_with_logits(pred_logit_shape, labels)
        focal_loss = self.focal_loss_fn(pred_logit_shape, labels)
        
        pred_prob = torch.sigmoid(pred_logit_shape)
        dice = dice_score(pred_prob, labels)
        dice_loss = 1.0 - dice
        
        shape_loss = 0.7 * dice_loss + 0.3 * focal_loss
        l2_loss = latent_l2_penalty(encoded)
        total_loss = shape_loss + self.l2_penalty_weight * l2_loss
        
        self.log('val/total_loss', total_loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('val/shape_loss', shape_loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('val/dice_loss', dice_loss, on_epoch=True, sync_dist=True)
        self.log('val/focal_loss', focal_loss, on_epoch=True, sync_dist=True)
        self.log('val/bce_loss', bce_loss, on_epoch=True, sync_dist=True)
        self.log('val/l2_loss', l2_loss, on_epoch=True, sync_dist=True)
        self.log('val/dice_score', dice, prog_bar=True, on_epoch=True, sync_dist=True)
        
        return total_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        if self.use_cosine_schedule:
            if self.total_steps is None:
                self.total_steps = self.trainer.estimated_stepping_batches
            
            warmup_steps = int(self.total_steps * self.warmup_ratio)
            
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps
                else:
                    progress = (step - warmup_steps) / (self.total_steps - warmup_steps)
                    return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)))
            
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            return {{
                'optimizer': optimizer,
                'lr_scheduler': {{
                    'scheduler': scheduler,
                    'interval': 'step',
                }}
            }}
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=self.milestones, gamma=self.gamma
            )
            return {{
                'optimizer': optimizer,
                'lr_scheduler': {{
                    'scheduler': scheduler,
                    'interval': 'epoch',
                }}
            }}
"""

train_script_template = """\"\"\"PyTorch Lightning trainer for {class_name} (stage1) NeAR.
\"\"\"
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

from lightning_module import {class_name}NeARLightningModule
from near.datasets.cardiac_dataset import CardiacClassDatasetWithBiasedSampling
from near.models.nn3d.grid import GatherGridsFromVolumes


class {class_name}DataModule:
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
    config_name = args.config if hasattr(args, 'config') and args.config else 'config_{class_name}'
    cfg_module = importlib.import_module(config_name)
    cfg = cfg_module.cfg
    
    print(f"\\n{{'='*70}}")
    print(f"Loading config from: {{config_name}}")
    print(f"Training epochs: {{cfg['n_epochs']}}")
    print(f"{{'='*70}}\\n")

    cfg['run_flag'] += time.strftime("%y%m%d_%H%M%S")
    base_path = os.path.join(cfg['base_path'], cfg['run_flag'])
    os.makedirs(base_path, exist_ok=True)

    dm = {class_name}DataModule(cfg)
    dm.setup()

    n_samples = len(dm.train_dataset)
    pl_module = {class_name}NeARLightningModule(
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
                print(f"Loaded pretrained weights into pl_module.model from {{resume_ckpt}}")
            except Exception:
                pl_module.load_state_dict(state, strict=False)
                print(f"Loaded checkpoint into pl_module (partial) from {{resume_ckpt}}")
        except Exception as e:
            print(f"Warning: failed to load resume checkpoint: {{e}}")

    wandb_logger = WandbLogger(project='NeAR_stage1_{class_name}', name=cfg['run_flag'])

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

    print(f"\\nTraining completed! Best checkpoint saved to: {{base_path}}/best.ckpt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--devices', type=int, default=1, help='Number of devices (GPUs) to use')
    parser.add_argument('--strategy', type=str, default=None, help='DDP strategy (e.g., ddp)')
    parser.add_argument('--config', type=str, default='config_{class_name}', help='Config module name (without .py)')
    args = parser.parse_args()

    main(args)
"""

run_job_template = """#!/bin/bash
#SBATCH -A project_2016517
#SBATCH -p gpu
#SBATCH --gres=gpu:v100:4
#SBATCH --mem=64G
#SBATCH -t 72:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --cpus-per-task=8
#SBATCH -J {class_name}_train

JOB_NAME="{class_name}_train" 
PY_FILE="{base_dir}/stage1_{class_name}/train_{class_name}_pl.py"

echo "==== SLURM JOB INFO ===="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on nodes: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "========================"

### === 激活你的环境 ===
module load python-data/3.10-24.04
source /projappl/project_2016517/chengjun/junjieenv/bin/activate

### === 切换到你的代码目录 ===
cd {base_dir}/stage1_{class_name}

echo "Current working directory: $(pwd)"
echo "Starting training..."

### === 运行你的 Python 代码 ===
python ${{PY_FILE}} --devices 4 --strategy ddp --config config_{class_name}

echo "Training finished."
"""

for class_name, class_index in classes:
    print(f"Processing {class_name} (Index: {class_index})...")
    
    # Create directory
    dir_name = f"stage1_{class_name}"
    dir_path = os.path.join(base_dir, dir_name)
    os.makedirs(dir_path, exist_ok=True)
    
    # Create config file
    with open(os.path.join(dir_path, f"config_{class_name}.py"), "w") as f:
        f.write(config_template.format(class_name=class_name, class_index=class_index))
        
    # Create lightning module
    with open(os.path.join(dir_path, "lightning_module.py"), "w") as f:
        f.write(lightning_module_template.format(class_name=class_name))
        
    # Create training script
    with open(os.path.join(dir_path, f"train_{class_name}_pl.py"), "w") as f:
        f.write(train_script_template.format(class_name=class_name))
        
    # Create run job script
    with open(os.path.join(dir_path, "run_job.sh"), "w") as f:
        f.write(run_job_template.format(class_name=class_name, base_dir=base_dir))
        
    print(f"Done for {class_name}.")

print("All classes processed.")
