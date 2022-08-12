import _init_paths_local
import fire
import time
import pandas as pd
import os
import importlib

import numpy as np
import torch

from collections import OrderedDict
from torch.utils.data import DataLoader

from utils.utils import MultiAverageMeter, to_var, to_device, categorical_to_one_hot, model_to_syncbn
from utils.metrics import cal_batch_iou, cal_batch_dice
from utils.loss import soft_dice_loss

from models.resnet import FCNResNet
from models.acsunet import ACSUNet

from acsconv.converters import Conv3dConverter

from near.datasets.seg_dataset import AlanSegDataset


def main():

    cfg = importlib.import_module("config_eval").cfg

    # Datasets
    train_set = AlanSegDataset(root=cfg['data_path'], resolution=cfg["resolution"])

    # Seg-FCN
    model = FCNResNet(pretrained=False, num_classes=2, backbone='resnet18')
    model = model_to_syncbn(Conv3dConverter(model, i3d_repeat_axis=-3))

    # Seg-UNet
    # model = ACSUNet(num_classes=2)

    model.load_state_dict(torch.load(cfg["model_path"], map_location=torch.device('cpu')), strict=True)

    val_loader = DataLoader(train_set, batch_size=1, shuffle=False)

    model = to_device(model)
    
    logs = ['loss', 'iou', 'dice', 'iou0', 'iou1', 'dice0', 'dice1', 'dice_global']
    val_logs = ['val_'+log for log in logs]

    log_dict = OrderedDict.fromkeys(val_logs, 0)

    eval_meters = test_epoch(
        model=model,
        loader=val_loader,
        cfg=cfg, 
        save_dir=cfg["save_dir"],
        is_test=False
    )

    for i, key in enumerate(val_logs):
        log_dict[key] = eval_meters[i]
        print('%s: %.4f' % (key, eval_meters[i]))


def test_epoch(model, loader, cfg, save_dir, print_freq=1, is_test=True):

    idx = 0
    df = pd.read_csv(os.path.join(cfg["data_path"], 'info.csv'))
    info = df[df['low_quality'].isnull()]
    info = info[['ROI_id', 'ROI_anomaly']]
    info.reset_index(drop=True, inplace=True)

    meters = MultiAverageMeter()
    model.eval()
    intersection = 0
    union = 0
    end = time.time()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            x = to_var(x)
            y = to_var(y)
            # forward
            pred_logit = model(x)
            # calculate metrics
            y_one_hot = categorical_to_one_hot(y, dim=1, expand_dim=False)
            pred_classes = pred_logit.argmax(1)

            # save predicted segmentation
            name = info.iloc[idx]['ROI_id']
            idx += 1
            np.save(os.path.join(save_dir, name), pred_classes.cpu().numpy())

            intersection += ((pred_classes==1) * (y[:,0]==1)).sum().item()
            union += ((pred_classes==1).sum() + y[:,0].sum()).item()

            loss = soft_dice_loss(pred_logit, y_one_hot)
            batch_size = y.size(0)

            
            iou = cal_batch_iou(pred_logit, y_one_hot)
            dice = cal_batch_dice(pred_logit, y_one_hot)

            logs = [loss.item(), iou[1:].mean(), dice[1:].mean()]+ \
                                [iou[i].item() for i in range(len(iou))]+ \
                                [dice[i].item() for i in range(len(dice))]+ \
                                [time.time() - end]
            meters.update(logs, batch_size)   

            end = time.time()

            print_freq = 2 // meters.val[-1] + 1
            if batch_idx % print_freq == 0:
                res = '\t'.join([
                    'Test' if is_test else 'Valid',
                    'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
                    'Time %.3f (%.3f)' % (meters.val[-1], meters.avg[-1]),
                    'Loss %.4f (%.4f)' % (meters.val[0], meters.avg[0]),
                    'IOU %.4f (%.4f)' % (meters.val[1], meters.avg[1]),
                    'DICE %.4f (%.4f)' % (meters.val[2], meters.avg[2]),
                ])
                print(res)
    dice_global = 2. * intersection / union

    return meters.avg[:-1] + [dice_global]

if __name__ == '__main__':
    fire.Fire(main)
