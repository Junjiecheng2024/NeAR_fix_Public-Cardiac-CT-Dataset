"""
PyTorch Lightning Module for NeAR v2.0 Phase1 Training (Shape + Appearance)
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Any, Optional
import numpy as np

from near.models.nn3d.model_shape_appearance import EmbeddingDecoderShapeAppearanceWithContext
from near.models.losses import latent_l2_penalty, dice_score, FocalLoss, BoundaryDiceLoss


class CoronaryTier2LightningModule(pl.LightningModule):
    """
    PyTorch Lightning wrapper for Coronary Tier2 training with Shape + Appearance.
    """
    
    def __init__(
        self,
        n_samples: int,
        grid_gather_fn,
        latent_dimension: int = 256,
        decoder_channels: list = [64, 48, 32, 16],
        appearance_channels: int = 64,
        use_context: bool = True,
        lr: float = 5e-4,
        l2_penalty_weight: float = 1e-4,
        dice_weight: float = 0.6,
        boundary_dice_weight: float = 0.2,
        focal_weight: float = 0.15,
        use_cosine_schedule: bool = True,
        warmup_ratio: float = 0.02,
        total_steps: Optional[int] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['grid_gather_fn'])
        
        self.grid_gather_fn = grid_gather_fn
        
        # Build model
        self.model = EmbeddingDecoderShapeAppearanceWithContext(
            latent_dimension=latent_dimension,
            n_samples=n_samples,
            decoder_channels=decoder_channels,
            appearance_channels=appearance_channels,
            use_context=use_context
        )
        
        # Loss functions
        self.focal_loss_fn = FocalLoss(alpha=0.25, gamma=4.0)
        self.boundary_dice_fn = BoundaryDiceLoss(boundary_width=2)
        
        # Loss weights
        self.dice_weight = dice_weight
        self.boundary_dice_weight = boundary_dice_weight
        self.focal_weight = focal_weight
        self.l2_penalty_weight = l2_penalty_weight
        
        # Optimizer params
        self.lr = lr
        self.use_cosine_schedule = use_cosine_schedule
        self.warmup_ratio = warmup_ratio
        self.total_steps = total_steps
    
    def forward(self, indices, grid, appearance, context=None):
        return self.model(indices, grid, appearance, context)
    
    def training_step(self, batch, batch_idx):
        # Unpack batch (dict from CoronaryTier2Dataset)
        indices = batch["index"]
        shape = batch["shape"]
        appearance = batch["appearance"]
        context = batch.get("context", None)
        
        # Generate sampling grid
        _, grids, labels = self.grid_gather_fn(shape)
        grids = grids.to(self.device)
        labels = labels.to(self.device)
        appearance = appearance.to(self.device)
        if context is not None:
            context = context.to(self.device)
        
        # Forward pass
        pred_logit, encoded = self(indices, grids, appearance, context)
        
        # Compute losses
        # 1. Dice loss
        pred_prob = torch.sigmoid(pred_logit)
        dice = dice_score(pred_prob, labels)
        dice_loss = 1.0 - dice
        
        # 2. Boundary Dice loss
        boundary_dice_loss = self.boundary_dice_fn(pred_prob, labels)
        
        # 3. Focal loss
        focal_loss = self.focal_loss_fn(pred_logit, labels)
        
        # 4. L2 regularization on latent
        l2_loss = latent_l2_penalty(encoded)
        
        # Combined loss
        shape_loss = (self.dice_weight * dice_loss + 
                     self.boundary_dice_weight * boundary_dice_loss +
                     self.focal_weight * focal_loss)
        total_loss = shape_loss + self.l2_penalty_weight * l2_loss
        
        # Logging
        self.log('train/total_loss', total_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train/dice_loss', dice_loss, on_step=True, on_epoch=True)
        self.log('train/boundary_dice_loss', boundary_dice_loss, on_step=True, on_epoch=True)
        self.log('train/focal_loss', focal_loss, on_step=True, on_epoch=True)
        self.log('train/dice_score', dice, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train/l2_loss', l2_loss, on_step=True, on_epoch=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        indices = batch["index"]
        shape = batch["shape"]
        appearance = batch["appearance"]
        context = batch.get("context", None)
        
        _, grids, labels = self.grid_gather_fn(shape)
        grids = grids.to(self.device)
        labels = labels.to(self.device)
        appearance = appearance.to(self.device)
        if context is not None:
            context = context.to(self.device)
        
        pred_logit, encoded = self(indices, grids, appearance, context)
        
        pred_prob = torch.sigmoid(pred_logit)
        dice = dice_score(pred_prob, labels)
        dice_loss = 1.0 - dice
        boundary_dice_loss = self.boundary_dice_fn(pred_prob, labels)
        focal_loss = self.focal_loss_fn(pred_logit, labels)
        l2_loss = latent_l2_penalty(encoded)
        
        shape_loss = (self.dice_weight * dice_loss + 
                     self.boundary_dice_weight * boundary_dice_loss +
                     self.focal_weight * focal_loss)
        total_loss = shape_loss + self.l2_penalty_weight * l2_loss
        
        self.log('val/total_loss', total_loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('val/dice_score', dice, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('val/dice_loss', dice_loss, on_epoch=True, sync_dist=True)
        self.log('val/boundary_dice_loss', boundary_dice_loss, on_epoch=True, sync_dist=True)
        
        return total_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5)
        
        if self.use_cosine_schedule:
            if self.total_steps is None:
                self.total_steps = self.trainer.estimated_stepping_batches
            
            warmup_steps = int(self.total_steps * self.warmup_ratio)
            
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / max(warmup_steps, 1)
                else:
                    progress = (step - warmup_steps) / max(self.total_steps - warmup_steps, 1)
                    return 0.5 * (1.0 + np.cos(np.pi * progress))
            
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
                optimizer, milestones=[100, 200, 300], gamma=0.5
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                }
            }
