"""
原始NeAR修复脚本（多类、带appearance）

功能说明：
1. 这是原始NeAR论文的训练脚本，支持多类+appearance模式
2. 数据集：AlanDataset / AbdomenCTDataset（非心脏数据）
3. 模型：EmbeddingDecoder（包含appearance分支）
4. 与阶段一的区别：
   - 阶段一：单类 + shape-only + 心脏冠状动脉
   - 此脚本：多类 + appearance + 腹部器官

用途：参考原始实现，不直接用于当前项目

Original NeAR repair script for multi-class with appearance.
"""

import _init_paths_local

import os
import time
import importlib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from near.datasets.refine_dataset import AlanDataset, AbdomenCTDataset, AbdomenCTDistortedDataset
from near.models.nn3d.grid import GatherGridsFromVolumes
from near.models.nn3d.model import EmbeddingDecoder
from near.models.losses import latent_l2_penalty, dice_score
from near.utils.misc import to_device, to_var, write_json, Metrics, surface_dice, cal_surface_dice


def setup_cfg(cfg):

    cfg["run_flag"] += time.strftime("%y%m%d_%H%M%S")

    base_path = os.path.join(cfg["base_path"], cfg["run_flag"])
    if os.path.exists(base_path):
        raise ValueError(
            "Existing [base_path]: %s! Use another `run_flag`. " % base_path)
    else:
        os.makedirs(base_path)

    write_json(cfg, os.path.join(base_path, "config.json"), verbose=True)

    return cfg, base_path


def train_epoch(model, optimizer, loader,
                shape_loss_fn, gather_fn,
                metrics_per_batch, metrics_per_epoch,
                l2_penalty_weight, writer, epoch):
    model.train()

    global iteration

    tmp_metrics = Metrics(*metrics_per_epoch.keys)
    for _, (indices, shape, appearance) in enumerate(loader):
        indices = to_var(indices)
        _, grids, labels = gather_fn(shape)
        # Note there are grid noises during the training. 
        # Therefore, the `appearance_label` could not align with the `labels` (with sight shift), which could serve as a data augmentation.
        # However, if your images are high-res, this shift will be large. If so, you can align them by concatenating then split.
        _, _, appearance_label = gather_fn(appearance) 

        pred_logit_shape, encoded = model(indices, grids, appearance_label)

        shape_loss = shape_loss_fn(pred_logit_shape, labels) # shape loss
        dice_metric = dice_score(pred_logit_shape.sigmoid() > 0.5, labels)
        surface_dice_metric = cal_surface_dice(pred_logit_shape.sigmoid().detach().cpu().numpy() > 0.5, labels.cpu().numpy().astype(bool))
        l2_loss = latent_l2_penalty(encoded)

        loss = shape_loss + l2_penalty_weight*l2_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tmp_metrics.ordered_update(loss.item(), 
                                   shape_loss.item(),
                                   dice_metric.item(),
                                   surface_dice_metric.item(),
                                   l2_loss.item())

        metrics_per_batch.ordered_update(loss.item(), 
                                         shape_loss.item(),
                                         dice_metric.item(),
                                         surface_dice_metric.item(),
                                         l2_loss.item())

        writer.add_scalar('iter_loss', loss.item(), iteration)
        iteration += 1

    metrics_per_epoch.ordered_update(*tmp_metrics.ordered_mean())
    metrics_per_epoch.log_latest(header="Train Loss: ")

    keys = ['train_loss', 'train_shape_loss', 'train_dice', 'train_surface_dice', 'train_l2_loss']
    vals = tmp_metrics.ordered_mean()
    for key, val in zip(keys, vals):
        writer.add_scalar(key, val, epoch)


def eval_epoch(model, loader,
               shape_loss_fn,  
               gather_fn, metrics_per_epoch, base_path,
               l2_penalty_weight, writer, epoch):
    model.eval()

    best_loss, _ = metrics_per_epoch.find_best("shape")

    tmp_metrics = Metrics(*metrics_per_epoch.keys)
    for _, (indices, shape, appearance) in enumerate(loader):
        indices = to_var(indices)
        _, grids, labels = gather_fn(shape)

        _, _, appearance_label = gather_fn(appearance)

        with torch.no_grad():
            pred_logit_shape, encoded = model(indices, grids, appearance_label)

            shape_loss = shape_loss_fn(pred_logit_shape, labels) # shape loss
            dice_metric = dice_score(pred_logit_shape.sigmoid() > 0.5, labels)
            surface_dice_metric = cal_surface_dice(pred_logit_shape.sigmoid().cpu().numpy() > 0.5, labels.cpu().numpy().astype(bool))

            l2_loss = latent_l2_penalty(encoded)
           
            loss = shape_loss + l2_penalty_weight*l2_loss


        tmp_metrics.ordered_update(loss.item(), 
                                   shape_loss.item(), 
                                   dice_metric.item(),
                                   surface_dice_metric.item(),
                                   l2_loss.item())

    metrics_per_epoch.ordered_update(*tmp_metrics.ordered_mean())
    metrics_per_epoch.log_latest(header="Eval Loss: ")

    keys = ['val_loss', 'val_shape_loss', 'val_dice', 'val_surface_dice', 'val_l2_loss']
    vals = tmp_metrics.ordered_mean()
    for key, val in zip(keys, vals):
        writer.add_scalar(key, val, epoch)

    torch.save(model.state_dict(), os.path.join(base_path, "latest.pth"))

    tmp_best_loss = tmp_metrics.ordered_mean()[1]  # shape loss

    if tmp_best_loss < best_loss:
        torch.save(model.state_dict(), os.path.join(base_path, "best.pth"))
        print("======================================================================")
        print("Found a new best model! Loss: ", tmp_best_loss)
        print("======================================================================")


if __name__ == "__main__":

    cfg, base_path = setup_cfg(importlib.import_module("config_near").cfg)

    data_path = cfg['data_path']
    
    train_dataset = AlanDataset(root=data_path, resolution=cfg["target_resolution"], n_samples=cfg["n_training_samples"])
    eval_dataset = AlanDataset(root=data_path, resolution=cfg["target_resolution"], n_samples=cfg["n_training_samples"])


    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True,
                              pin_memory=(torch.cuda.is_available()), num_workers=cfg["n_workers"])
    train_gather_fn = GatherGridsFromVolumes(
        cfg["training_resolution"], grid_noise=cfg["grid_noise"], uniform_grid_noise=cfg["uniform_grid_noise"])


    eval_loader = DataLoader(train_dataset, batch_size=cfg["eval_batch_size"], shuffle=False,
                             pin_memory=(torch.cuda.is_available()), num_workers=cfg["n_workers"])
    eval_gather_fn = GatherGridsFromVolumes(
        cfg["target_resolution"], grid_noise=None)

    print('total data: ', len(train_dataset))


    training_metrics = Metrics("total", "shape", "dice", "surface_dice", "l2")
    train_metrics = Metrics("total", "shape", "dice", "surface_dice", "l2")
    eval_metrics = Metrics("total", "shape", "dice", "surface_dice", "l2")

    writer = SummaryWriter(log_dir=os.path.join(base_path, 'Tensorboard_Results'))

    # define model and optimization
    model = to_device(EmbeddingDecoder(n_samples=len(train_dataset), appearance=cfg['appearance']))

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

        
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg["milestones"], gamma=0.1)

    shape_loss_fn = nn.BCEWithLogitsLoss()
    l2_penalty_weight = cfg["l2_penalty_weight"]
   
    eval_epoch(model=model, loader=eval_loader,
               shape_loss_fn=shape_loss_fn, 
               gather_fn=eval_gather_fn,
               metrics_per_epoch=eval_metrics, base_path=base_path,
               l2_penalty_weight=l2_penalty_weight, 
               writer=writer, epoch=0)

    global iteration
    iteration = 0

    for epoch in range(cfg["n_epochs"]):
        print("Epoch", epoch+1)

        train_epoch(model=model, optimizer=optimizer,
                    loader=train_loader, shape_loss_fn=shape_loss_fn, 
                    gather_fn=train_gather_fn,
                    metrics_per_batch=training_metrics,
                    metrics_per_epoch=train_metrics,
                    l2_penalty_weight=l2_penalty_weight,
                    writer=writer, epoch=epoch)
        eval_epoch(model=model, loader=eval_loader,
                   shape_loss_fn=shape_loss_fn, 
                   gather_fn=eval_gather_fn,
                   metrics_per_epoch=eval_metrics, base_path=base_path,
                   l2_penalty_weight=l2_penalty_weight,
                   writer=writer, epoch=epoch)

        scheduler.step()

        training_metrics.save(os.path.join(base_path, "training_loss.json"))
        train_metrics.save(os.path.join(base_path, "train_loss.json"))
        eval_metrics.save(os.path.join(base_path, "eval_loss.json"))

    writer.close()
