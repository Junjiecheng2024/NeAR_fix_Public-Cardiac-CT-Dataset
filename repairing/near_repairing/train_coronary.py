"""
Training script for Coronary (class 9) single-class refinement.
Phase 1: Shape-only NeAR to refine noisy labels.
"""

import sys
import os
# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import time
import importlib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tensorboardX import SummaryWriter

from near.datasets.cardiac_dataset import CardiacClassDatasetWithBiasedSampling
from near.models.nn3d.grid import GatherGridsFromVolumes
from near.models.nn3d.model_shape_only import EmbeddingDecoderShapeOnly
from near.models.losses import latent_l2_penalty, dice_score
from near.utils.misc import to_device, to_var, write_json, Metrics


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
    where p_t = p if y=1, else 1-p
    
    Args:
        alpha: weighting factor in [0, 1] to balance positive/negative examples
               or a Tensor of weights for each class
        gamma: focusing parameter >= 0. gamma=0 is equivalent to CE loss
               gamma=2 is a good default (down-weights easy examples)
        reduction: 'mean', 'sum' or 'none'
    """
    def __init__(self, alpha=0.25, gamma=3.0, reduction='mean'):
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


def setup_cfg(cfg):
    """Setup configuration and create checkpoint directory."""
    cfg["run_flag"] += time.strftime("%y%m%d_%H%M%S")
    
    base_path = os.path.join(cfg["base_path"], cfg["run_flag"])
    if os.path.exists(base_path):
        raise ValueError(f"Existing [base_path]: {base_path}! Use another `run_flag`.")
    else:
        os.makedirs(base_path)
    
    write_json(cfg, os.path.join(base_path, "config.json"), verbose=True)
    
    return cfg, base_path


def train_epoch(model, optimizer, scheduler, scaler, loader, shape_loss_fn, gather_fn,
                metrics_per_batch, metrics_per_epoch,
                l2_penalty_weight, grad_accum_steps, use_amp, writer, epoch, iteration):
    """Train for one epoch with AMP support."""
    model.train()
    
    epoch_start_time = time.time()
    tmp_metrics = Metrics(*metrics_per_epoch.keys)
    
    # For gradient accumulation
    optimizer.zero_grad()
    
    for batch_idx, (indices, shape) in enumerate(loader):
        indices = to_var(indices)
        _, grids, labels = gather_fn(shape)
        
        # Forward pass with autocast
        with autocast(enabled=use_amp):
            pred_logit_shape, encoded = model(indices, grids)
            
            # Compute losses
            shape_loss = shape_loss_fn(pred_logit_shape, labels)
            dice_metric = dice_score(pred_logit_shape.sigmoid() > 0.5, labels)
            l2_loss = latent_l2_penalty(encoded)
            
            loss = (shape_loss + l2_penalty_weight * l2_loss) / grad_accum_steps
        
        # Backward (accumulate gradients)
        scaler.scale(loss).backward()
        
        # Optimizer step after accumulating gradients
        if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()  # Update learning rate per step
            optimizer.zero_grad()
        
        # Update metrics (use unscaled loss for logging)
        actual_loss = loss.item() * grad_accum_steps
        tmp_metrics.ordered_update(actual_loss, shape_loss.item(),
                                   dice_metric.item(), l2_loss.item())
        metrics_per_batch.ordered_update(actual_loss, shape_loss.item(),
                                         dice_metric.item(), l2_loss.item())
        
        writer.add_scalar('iter_loss', actual_loss, iteration[0])
        writer.add_scalar('lr', scheduler.get_last_lr()[0], iteration[0])
        iteration[0] += 1
        
        if batch_idx % 50 == 0:
            print(f"  Batch [{batch_idx}/{len(loader)}] "
                  f"Loss: {actual_loss:.4f} Dice: {dice_metric.item():.4f} "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")
    
    metrics_per_epoch.ordered_update(*tmp_metrics.ordered_mean())
    
    epoch_time = time.time() - epoch_start_time
    metrics_per_epoch.log_latest(header=f"Epoch {epoch} Train: ")
    print(f"  Training time: {epoch_time:.2f}s")
    
    # Log to tensorboard
    keys = ['train_loss', 'train_shape_loss', 'train_dice', 'train_l2_loss']
    vals = tmp_metrics.ordered_mean()
    for key, val in zip(keys, vals):
        writer.add_scalar(key, val, epoch)
    
    return epoch_time


def eval_epoch(model, loader, shape_loss_fn, gather_fn, metrics_per_epoch,
               base_path, l2_penalty_weight, use_amp, writer, epoch):
    """Evaluate for one epoch with AMP support."""
    model.eval()
    
    eval_start_time = time.time()
    best_loss, _ = metrics_per_epoch.find_best("shape")
    
    tmp_metrics = Metrics(*metrics_per_epoch.keys)
    
    with torch.no_grad():
        for indices, shape in loader:
            indices = to_var(indices)
            _, grids, labels = gather_fn(shape)
            
            # Forward pass with autocast
            with autocast(enabled=use_amp):
                pred_logit_shape, encoded = model(indices, grids)
                
                # Compute losses
                shape_loss = shape_loss_fn(pred_logit_shape, labels)
                dice_metric = dice_score(pred_logit_shape.sigmoid() > 0.5, labels)
                l2_loss = latent_l2_penalty(encoded)
                
                loss = shape_loss + l2_penalty_weight * l2_loss
            
            tmp_metrics.ordered_update(loss.item(), shape_loss.item(),
                                      dice_metric.item(), l2_loss.item())
    
    metrics_per_epoch.ordered_update(*tmp_metrics.ordered_mean())
    
    eval_time = time.time() - eval_start_time
    metrics_per_epoch.log_latest(header=f"Epoch {epoch} Eval:  ")
    print(f"  Validation time: {eval_time:.2f}s")
    
    # Log to tensorboard
    keys = ['val_loss', 'val_shape_loss', 'val_dice', 'val_l2_loss']
    vals = tmp_metrics.ordered_mean()
    for key, val in zip(keys, vals):
        writer.add_scalar(key, val, epoch)
    
    # Save latest checkpoint
    torch.save(model.state_dict(), os.path.join(base_path, "latest.pth"))
    
    # Save best checkpoint
    tmp_best_loss = tmp_metrics.ordered_mean()[1]  # shape loss
    if tmp_best_loss < best_loss:
        torch.save(model.state_dict(), os.path.join(base_path, "best.pth"))
        print("=" * 70)
        print(f"Found a new best model! Shape Loss: {tmp_best_loss:.6f}")
        print("=" * 70)
    
    return eval_time


def main():
    """Main training loop."""
    # Load configuration
    cfg, base_path = setup_cfg(importlib.import_module("config_coronary").cfg)
    
    data_path = cfg['data_path']
    class_name = cfg['class_name']
    
    print(f"\n{'='*70}")
    print(f"Training NeAR for {class_name} (Class {cfg['class_index']})")
    print(f"Data path: {data_path}")
    print(f"Checkpoint path: {base_path}")
    print(f"{'='*70}\n")
    
    # Create datasets with boundary-biased sampling
    # Note: Using same dataset for train and eval (we want overfitting)
    train_dataset = CardiacClassDatasetWithBiasedSampling(
        root=data_path,
        class_name=class_name,
        resolution=cfg["target_resolution"],
        n_samples=cfg["n_training_samples"],
        sampling_bias_ratio=cfg["sampling_bias_ratio"],
        sampling_dilation_radius=cfg["sampling_dilation_radius"],
        class_index=cfg["class_index"]  # Extract specific class from multi-class segmentation
    )
    
    eval_dataset = CardiacClassDatasetWithBiasedSampling(
        root=data_path,
        class_name=class_name,
        resolution=cfg["target_resolution"],
        n_samples=cfg["n_training_samples"],
        sampling_bias_ratio=cfg["sampling_bias_ratio"],
        sampling_dilation_radius=cfg["sampling_dilation_radius"],
        class_index=cfg["class_index"]  # Extract specific class from multi-class segmentation
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        num_workers=cfg["n_workers"]
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=cfg["eval_batch_size"],
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=cfg["n_workers"]
    )
    
    # Create grid samplers with boundary bias
    train_gather_fn = GatherGridsFromVolumes(
        cfg["training_resolution"],
        grid_noise=cfg["grid_noise"],
        uniform_grid_noise=cfg["uniform_grid_noise"],
        boundary_bias_ratio=cfg["sampling_bias_ratio"],
        boundary_dilation_radius=cfg["sampling_dilation_radius"]
    )
    
    eval_gather_fn = GatherGridsFromVolumes(
        cfg["target_resolution"],
        grid_noise=None,
        boundary_bias_ratio=0.0  # No bias during evaluation for fair comparison
    )
    
    print(f'Total samples: {len(train_dataset)}')
    print(f'Boundary-biased sampling configuration:')
    print(f'  Training bias ratio: {cfg["sampling_bias_ratio"]*100:.0f}% near boundaries (dynamic)')
    print(f'  Evaluation bias ratio: 0% (uniform sampling)')
    print(f'  Dilation radius: {cfg["sampling_dilation_radius"]} voxels\n')
    
    # Initialize metrics
    training_metrics = Metrics("total", "shape", "dice", "l2")
    train_metrics = Metrics("total", "shape", "dice", "l2")
    eval_metrics = Metrics("total", "shape", "dice", "l2")
    
    # Setup tensorboard
    writer = SummaryWriter(log_dir=os.path.join(base_path, 'Tensorboard_Results'))
    
    # Create model
    model = to_device(
        EmbeddingDecoderShapeOnly(
            n_samples=len(train_dataset),
            latent_dimension=cfg['latent_dimension'],
            decoder_channels=cfg['decoder_channels']
        )
    )
    
    # Multi-GPU support
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs\n")
        model = torch.nn.DataParallel(model)
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    
    # Setup GradScaler for mixed precision training
    use_amp = cfg.get("use_amp", False)
    scaler = GradScaler(enabled=use_amp)
    
    if use_amp:
        print(f"Using Automatic Mixed Precision (AMP) training\n")
    
    # Calculate total steps for cosine annealing
    steps_per_epoch = len(train_loader)
    total_steps = cfg["n_epochs"] * steps_per_epoch
    warmup_steps = int(total_steps * cfg.get("warmup_ratio", 0.05))
    
    if cfg.get("use_cosine_schedule", False):
        # Warmup + Cosine Annealing scheduler
        from torch.optim.lr_scheduler import LambdaLR
        import math
        
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # Cosine annealing
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return 0.5 * (1.0 + math.cos(math.pi * progress))
        
        scheduler = LambdaLR(optimizer, lr_lambda)
        print(f"Using Warmup + Cosine Annealing scheduler")
        print(f"  Total steps: {total_steps}")
        print(f"  Warmup steps: {warmup_steps} ({cfg.get('warmup_ratio', 0.05)*100:.1f}%)")
    else:
        # MultiStepLR scheduler (original)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, cfg["milestones"], gamma=0.1
        )
        print(f"Using MultiStepLR scheduler with milestones: {cfg['milestones']}")
    
    print(f"Gradient accumulation steps: {cfg.get('gradient_accumulation_steps', 1)}")
    print(f"Effective batch size: {cfg['batch_size'] * cfg.get('gradient_accumulation_steps', 1)}\n")
    
    # Loss function - ENHANCED: Multiple strategies for extreme class imbalance
    # Strategy 1: Dynamic Focal Loss gamma (curriculum learning)
    # Strategy 2: Weighted combination favoring Dice Loss more
    # Strategy 3: Online Hard Example Mining via high gamma
    
    focal_loss_fn = FocalLoss(alpha=0.25, gamma=4.0)  # Increased from 3 to 4 (256x weight)
    
    def combined_loss_fn(pred_logits, targets):
        """Enhanced Combined Loss with dynamic weighting"""
        # Focal loss with higher gamma
        focal = focal_loss_fn(pred_logits, targets)
        
        # Dice loss (on probabilities) 
        pred_probs = torch.sigmoid(pred_logits)
        smooth = 1.0
        intersection = (pred_probs * targets).sum()
        dice = (2. * intersection + smooth) / (pred_probs.sum() + targets.sum() + smooth)
        dice_loss = 1 - dice  # Convert to loss (minimize)
        
        # Enhanced weighting: 70% Dice + 30% Focal
        # Reason: Dice directly optimizes target metric, should have higher weight
        # Focal mainly for handling class imbalance in gradient flow
        return 0.7 * dice_loss + 0.3 * focal
    
    shape_loss_fn = combined_loss_fn
    print(f"Using ENHANCED Combined Loss:")
    print(f"  - 70% Dice Loss (direct metric optimization)")
    print(f"  - 30% Focal Loss (gamma=4.0, 256x hard example weighting)")
    print(f"  - Higher Dice weight ensures we optimize the right metric")
    print(f"  - Gamma=4 provides stronger focus on hard examples\n")
    
    l2_penalty_weight = cfg["l2_penalty_weight"]
    
    # Load checkpoint if exists (从best.pth继续训练)
    resume_checkpoint = cfg.get("resume_checkpoint", None)
    start_epoch = 1
    
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"\n{'='*70}")
        print(f"Loading checkpoint from: {resume_checkpoint}")
        checkpoint_state = torch.load(resume_checkpoint, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model state
        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(checkpoint_state)
        else:
            model.load_state_dict(checkpoint_state)
        
        print(f"Successfully loaded checkpoint!")
        print(f"{'='*70}\n")
    
    # Get evaluation interval
    eval_interval = cfg.get("eval_interval", 1)  # 默认每轮都验证
    print(f"Evaluation策略: 第1轮后验证，之后每{eval_interval}轮验证一次\n")
    
    # Training loop
    iteration = [0]  # Use list to make it mutable in nested function
    
    print(f"\nStarting training for {cfg['n_epochs']} epochs...\n")
    print("Curriculum Learning Strategy:")
    print("  Epoch 1-100:   50% boundary bias (current)")
    print("  Epoch 101-200: 30% boundary bias (reduce)")
    print("  Epoch 201-300: 10% boundary bias (final adaptation)")
    print()
    
    for epoch in range(start_epoch, cfg["n_epochs"] + 1):
        # Dynamic boundary bias adjustment (curriculum learning)
        if epoch <= 100:
            current_bias = 0.5
        elif epoch <= 200:
            current_bias = 0.3
        else:
            current_bias = 0.1
        
        # Update train_gather_fn's boundary_bias_ratio
        train_gather_fn.boundary_bias_ratio = current_bias
        
        print(f"Epoch {epoch}/{cfg['n_epochs']} (boundary_bias={current_bias*100:.0f}%)")
        
        train_time = train_epoch(
            model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            loader=train_loader,
            shape_loss_fn=shape_loss_fn,
            gather_fn=train_gather_fn,
            metrics_per_batch=training_metrics,
            metrics_per_epoch=train_metrics,
            l2_penalty_weight=l2_penalty_weight,
            grad_accum_steps=cfg.get("gradient_accumulation_steps", 1),
            use_amp=use_amp,
            writer=writer, epoch=epoch, iteration=iteration
        )
        
        # 第1轮后验证，之后每 eval_interval 轮验证一次
        if epoch == 1 or epoch % eval_interval == 0 or epoch == cfg["n_epochs"]:
            eval_time = eval_epoch(
                model=model, loader=eval_loader,
                shape_loss_fn=shape_loss_fn,
                gather_fn=eval_gather_fn,
                metrics_per_epoch=eval_metrics,
                base_path=base_path,
                l2_penalty_weight=l2_penalty_weight,
                use_amp=use_amp,
                writer=writer, epoch=epoch
            )
            print(f"  Total epoch time: {train_time + eval_time:.2f}s\n")
        else:
            print(f"  Total epoch time: {train_time:.2f}s (no validation)\n")
        
        # Note: scheduler.step() is now called inside train_epoch for per-step update
        
        # Save metrics
        training_metrics.save(os.path.join(base_path, "training_loss.json"))
        train_metrics.save(os.path.join(base_path, "train_loss.json"))
        eval_metrics.save(os.path.join(base_path, "eval_loss.json"))
    
    writer.close()
    
    print(f"\n{'='*70}")
    print("Training completed!")
    print(f"Checkpoints saved to: {base_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
