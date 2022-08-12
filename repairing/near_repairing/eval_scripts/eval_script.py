import _init_paths_local

import os
import time
import importlib
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from near.datasets.refine_dataset import AlanDataset
from near.models.nn3d.grid import GatherGridsFromVolumes
from near.models.nn3d.model import EmbeddingDecoder
from near.models.losses import latent_l2_penalty, dice_score
from near.utils.misc import to_device, to_var, Metrics, load_model_on_multi_gpu, load_model_on_single_gpu

from near.utils.plot3d import plotly_3d_scan_to_html


def setup_cfg(cfg):
    base_path = cfg["base_path"]
    return cfg, base_path


def eval(model, loader, shape_loss_fn, gather_fn, base_path, eval_metrics):
    model.eval()

    dice_ret = []

    vis_save_dir = os.path.join(base_path, 'vis_result')
    shape_save_dir = os.path.join(base_path, 'shape_res')

    appearance_save_dir = os.path.join(base_path, 'appearance_res')

    if not os.path.isdir(vis_save_dir):
        os.makedirs(vis_save_dir)
    if not os.path.isdir(shape_save_dir):
        os.makedirs(shape_save_dir)
    if not os.path.isdir(appearance_save_dir):
        os.makedirs(appearance_save_dir)

    for index, (indices, shape, appearance) in enumerate(loader):
        # print(indices)
        indices = to_var(indices)
        _, grids, labels = gather_fn(shape)
        _, _, appearance_label = gather_fn(appearance)

        with torch.no_grad():
            pred_logit_shape, encoded = model(indices, grids, appearance_label)

            shape_loss = shape_loss_fn(pred_logit_shape, labels) # shape loss

            dice_metric = dice_score(pred_logit_shape.sigmoid() > 0.5, labels)
            l2_loss = latent_l2_penalty(encoded)

            eval_metrics.ordered_update(shape_loss.item(), 
                                   dice_metric.item(),
                                   l2_loss.item())

            dice_ret.append(dice_metric.item())

            eval_metrics.log_latest(header="Eval Loss: ")

            output_shape = (pred_logit_shape.squeeze(0).squeeze(0).sigmoid() > 0.5).cpu().numpy()

            # plotly_3d_scan_to_html(output_shape, os.path.join(vis_save_dir, '%s_output.html' % (index)))
            np.save(os.path.join(shape_save_dir, '%s_output' % (index)), output_shape)

    print('dice: %.6f' % np.mean(dice_ret) )


if __name__ == "__main__":

    cfg, base_path = setup_cfg(importlib.import_module("config_eval").cfg)

    data_path = cfg['data_path']
    
    train_dataset = AlanDataset(root=data_path, resolution=cfg["target_resolution"], n_samples=cfg["n_evaluation_samples"])

    eval_loader = DataLoader(train_dataset, batch_size=cfg["eval_batch_size"], shuffle=False,
                             pin_memory=(torch.cuda.is_available()), num_workers=cfg["n_workers"])
    eval_gather_fn = GatherGridsFromVolumes(
        cfg["target_resolution"], grid_noise=None)

    eval_metrics = Metrics("shape", "dice", "l2")

    # define model and optimization
    model = to_device(EmbeddingDecoder(n_samples=len(train_dataset), appearance=cfg['appearance']))

    model_path = os.path.join(base_path, 'best.pth')

    model = load_model_on_multi_gpu(model, model_path, {"map_location":torch.device('cpu')})

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # shape loss
    shape_loss_fn = nn.BCEWithLogitsLoss()

    eval(model=model, loader=eval_loader,
                shape_loss_fn=shape_loss_fn, 
                gather_fn=eval_gather_fn,
                base_path=base_path,
                eval_metrics=eval_metrics)

    eval_metrics.save(os.path.join(base_path, "eval_loss.json"))
