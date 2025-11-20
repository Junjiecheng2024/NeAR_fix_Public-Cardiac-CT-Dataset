"""
PyTorch Lightning Module for Aorta NeAR Training
"""
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
    """
    Focal Loss for addressing class imbalance.
    """
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


class AortaNeARLightningModule(pl.LightningModule):
    """
    PyTorch Lightning wrapper for NeAR Aorta training.
    """
    
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
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                }
            }
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=self.milestones, gamma=self.gamma
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                }
            }
