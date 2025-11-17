"""
PyTorch Lightning Module for Coronary NeAR Training

功能说明：
1. 封装训练逻辑到 LightningModule，支持多GPU训练
2. 集成动态边界采样、组合损失函数、课程学习策略
3. 自动管理优化器、学习率调度、混合精度训练
4. 支持 WandB 日志和 checkpoint 管理
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Any, Optional

from near.models.nn3d.model_shape_only import EmbeddingDecoderShapeOnly
from near.models.losses import latent_l2_penalty, dice_score


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: weighting factor in [0, 1]
        gamma: focusing parameter >= 0
        reduction: 'mean', 'sum' or 'none'
    """
    def __init__(self, alpha=0.25, gamma=4.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, *) logits (before sigmoid)
            targets: (N, *) binary targets (0 or 1)
        """
        # Apply sigmoid to get probabilities
        p = torch.sigmoid(inputs)
        
        # Calculate CE loss
        ce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        
        # Calculate p_t
        p_t = p * targets + (1 - p) * (1 - targets)
        
        # Calculate focal term: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Calculate focal loss
        focal_loss = focal_weight * ce_loss
        
        # Apply alpha weighting
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * focal_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CoronaryNeARLightningModule(pl.LightningModule):
    """
    Lightning Module for Coronary NeAR training.
    
    Encapsulates:
    - Model architecture (EmbeddingDecoderShapeOnly)
    - Loss functions (70% Dice + 30% Focal)
    - Optimizer and scheduler configuration
    - Training and validation steps
    - Logging and metrics
    """
    
    def __init__(
        self,
        n_samples: int,
        latent_dimension: int = 256,
        decoder_channels: list = [64, 48, 32, 16],
        lr: float = 1e-3,
        l2_penalty_weight: float = 1e-4,
        dice_weight: float = 0.7,
        focal_weight: float = 0.3,
        focal_gamma: float = 4.0,
        focal_alpha: float = 0.25,
        use_cosine_schedule: bool = False,
        warmup_ratio: float = 0.01,
        total_steps: Optional[int] = None,
        milestones: list = [100, 200],
        gamma: float = 0.5,
        **kwargs
    ):
        """
        Args:
            n_samples: Number of training samples (for embedding layer)
            latent_dimension: Dimension of latent vectors
            decoder_channels: Channel configuration for decoder
            lr: Learning rate
            l2_penalty_weight: Weight for L2 regularization on latent codes
            dice_weight: Weight for Dice loss in combined loss
            focal_weight: Weight for Focal loss in combined loss
            focal_gamma: Gamma parameter for Focal loss
            focal_alpha: Alpha parameter for Focal loss
            use_cosine_schedule: Whether to use cosine annealing scheduler
            warmup_ratio: Ratio of warmup steps (for cosine scheduler)
            total_steps: Total training steps (for cosine scheduler)
            milestones: Milestones for MultiStepLR (if not using cosine)
            gamma: Learning rate decay factor for MultiStepLR
        """
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Model
        self.model = EmbeddingDecoderShapeOnly(
            n_samples=n_samples,
            latent_dimension=latent_dimension,
            decoder_channels=decoder_channels
        )
        
        # Loss functions
        self.focal_loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.l2_penalty_weight = l2_penalty_weight
        
        # Scheduler config (stored for configure_optimizers)
        self.lr = lr
        self.use_cosine_schedule = use_cosine_schedule
        self.warmup_ratio = warmup_ratio
        self.total_steps = total_steps
        self.milestones = milestones
        self.gamma = gamma
        
        # For tracking best validation loss
        self.best_val_loss = float('inf')
    
    def forward(self, indices, grids):
        """Forward pass through the model."""
        return self.model(indices, grids)
    
    def combined_loss_fn(self, pred_logits, targets):
        """
        Enhanced Combined Loss with dynamic weighting
        
        Args:
            pred_logits: (B, 1, D, H, W) predicted logits
            targets: (B, 1, D, H, W) ground truth labels
            
        Returns:
            loss: weighted combination of Dice and Focal loss
        """
        # Focal loss
        focal = self.focal_loss_fn(pred_logits, targets)
        
        # Dice loss (on probabilities)
        pred_probs = torch.sigmoid(pred_logits)
        smooth = 1.0
        intersection = (pred_probs * targets).sum()
        dice = (2. * intersection + smooth) / (pred_probs.sum() + targets.sum() + smooth)
        dice_loss = 1 - dice  # Convert to loss (minimize)
        
        # Weighted combination
        return self.dice_weight * dice_loss + self.focal_weight * focal
    
    def training_step(self, batch, batch_idx):
        """
        Training step.
        
        Args:
            batch: (indices, shape, grids, labels) from dataloader
            batch_idx: batch index
            
        Returns:
            loss: total loss for backpropagation
        """
        indices, grids, labels = batch
        
        # Forward pass
        pred_logit_shape, encoded = self(indices, grids)
        
        # Compute losses
        shape_loss = self.combined_loss_fn(pred_logit_shape, labels)
        dice_metric = dice_score(pred_logit_shape.sigmoid() > 0.5, labels)
        l2_loss = latent_l2_penalty(encoded)
        
        # Total loss
        loss = shape_loss + self.l2_penalty_weight * l2_loss
        
        # Logging
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train/shape_loss', shape_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train/dice', dice_metric, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train/l2_loss', l2_loss, on_step=True, on_epoch=True, sync_dist=True)
        
        # Log learning rate
        if self.trainer.global_step > 0:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('train/lr', lr, on_step=True, on_epoch=False, sync_dist=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step.
        
        Args:
            batch: (indices, shape, grids, labels) from dataloader
            batch_idx: batch index
            
        Returns:
            Dict containing validation metrics
        """
        indices, grids, labels = batch
        
        # Forward pass
        pred_logit_shape, encoded = self(indices, grids)
        
        # Compute losses
        shape_loss = self.combined_loss_fn(pred_logit_shape, labels)
        dice_metric = dice_score(pred_logit_shape.sigmoid() > 0.5, labels)
        l2_loss = latent_l2_penalty(encoded)
        
        # Total loss
        loss = shape_loss + self.l2_penalty_weight * l2_loss
        
        # Logging
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val/shape_loss', shape_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/dice', dice_metric, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val/l2_loss', l2_loss, on_step=False, on_epoch=True, sync_dist=True)
        
        return {'val_loss': loss, 'val_dice': dice_metric}
    
    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""
        # Get average validation loss
        avg_val_loss = self.trainer.callback_metrics.get('val/loss', None)
        
        if avg_val_loss is not None and avg_val_loss < self.best_val_loss:
            self.best_val_loss = avg_val_loss
            # Log best loss
            self.log('val/best_loss', self.best_val_loss, sync_dist=True)
    
    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.
        
        Returns:
            Dict containing optimizer and scheduler configuration
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        if self.use_cosine_schedule:
            # Cosine Annealing with warmup
            if self.total_steps is None:
                raise ValueError("total_steps must be provided for cosine scheduler")
            
            warmup_steps = int(self.total_steps * self.warmup_ratio)
            
            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    # Linear warmup
                    return float(current_step) / float(max(1, warmup_steps))
                else:
                    # Cosine annealing
                    import math
                    progress = float(current_step - warmup_steps) / float(max(1, self.total_steps - warmup_steps))
                    return 0.5 * (1.0 + math.cos(math.pi * progress))
            
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',  # Update per step
                    'frequency': 1
                }
            }
        else:
            # MultiStepLR
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=self.milestones, gamma=self.gamma
            )
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',  # Update per epoch
                    'frequency': 1
                }
            }
