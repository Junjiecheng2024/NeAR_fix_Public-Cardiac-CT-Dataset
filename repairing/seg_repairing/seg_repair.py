import _init_paths_local
import fire
import time
import importlib
import os
import torch

from collections import OrderedDict
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from utils.sync_batchnorm import DataParallelWithCallback
from utils.utils import MultiAverageMeter, log_results, to_var, to_device, categorical_to_one_hot, model_to_syncbn
from utils.metrics import cal_batch_iou, cal_batch_dice
from utils.loss import soft_dice_loss

from models.resnet import FCNResNet
from models.acsunet import ACSUNet
from acsconv.converters import Conv3dConverter

from near.datasets.seg_dataset import AlanSegDataset


def setup_cfg(cfg):
    
    cfg["run_flag"] += time.strftime("%y%m%d_%H%M%S")

    # setup path
    base_path = os.path.join(cfg["base_path"], cfg["run_flag"])
    if os.path.exists(base_path):
        raise ValueError(
            "Existing [base_path]: %s! Use another `run_flag`. " % base_path)
    else:
        os.makedirs(base_path)

    return cfg, base_path


def main():

    cfg, save_path = setup_cfg(importlib.import_module("config_seg").cfg)

    # Datasets
    train_set = AlanSegDataset(root=cfg["data_path"], resolution=cfg["resolution"])
    valid_set = AlanSegDataset(root=cfg["data_path"], resolution=cfg["resolution"])

    # Seg-UNet
    # model = ACSUNet(num_classes=2)

    # Seg-FCN
    model = FCNResNet(pretrained=False, num_classes=2, backbone='resnet18')
    model = model_to_syncbn(Conv3dConverter(model, i3d_repeat_axis=-3))

    print(model)
    torch.save(model.state_dict(), os.path.join(save_path, 'model.pth'))

    train(model=model, train_set=train_set, valid_set=valid_set, save=save_path, n_epochs=cfg["n_epochs"], cfg=cfg)

    print('Done!')


def train(model, train_set, valid_set, save, n_epochs, cfg):

    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True,
                                pin_memory=(torch.cuda.is_available()), num_workers=cfg["num_workers"])
    valid_loader = DataLoader(valid_set, batch_size=cfg["batch_size"], shuffle=False,
                                pin_memory=(torch.cuda.is_available()), num_workers=cfg["num_workers"])
 
    # Model on cuda
    model = to_device(model)
    # Wrap model for multi-GPUs, if necessary
    model_wrapper = model
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:       
        if cfg["use_syncbn"]:
            print('Using sync-bn')
            model_wrapper = DataParallelWithCallback(model).cuda()
        else:
            model_wrapper = torch.nn.DataParallel(model).cuda()
    # optimizer and scheduler
    optimizer = torch.optim.Adam(model_wrapper.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg["milestones"],
                                                     gamma=cfg["gamma"])
    # Start logging
    logs = ['loss', 'iou', 'dice', 'iou0', 'iou1', 'dice0', 'dice1', 'dice_global']
    train_logs = ['train_'+log for log in logs]
    val_logs = ['val_'+log for log in logs]

    log_dict = OrderedDict.fromkeys(train_logs+val_logs, 0)
    with open(os.path.join(save, 'logs.csv'), 'w') as f:
        f.write('epoch,')
        for key in log_dict.keys():
            f.write(key+',')
        f.write('\n')
    with open(os.path.join(save, 'loss_logs.csv'), 'w') as f:
        f.write('iter,train_loss,\n')
    writer = SummaryWriter(log_dir=os.path.join(save, 'Tensorboard_Results'))

    # train and test the model
    best_dice_global = 0
    dice_to_noise = 0
    global iteration
    iteration = 0
    for epoch in range(n_epochs):
        print('\nlearning rate: ', scheduler.get_lr())
        # train epoch
        train_meters = train_epoch(
            model=model_wrapper,
            loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            n_epochs=n_epochs,
            cfg=cfg, 
            writer=writer
        )
        # eval epoch
        eval_meters = test_epoch(
            model=model_wrapper,
            loader=valid_loader,
            is_test=False
        )
        scheduler.step()


        # Log results
        for i, key in enumerate(train_logs):
            log_dict[key] = train_meters[i]
        for i, key in enumerate(val_logs):
            log_dict[key] = eval_meters[i]
        log_results(save, epoch, log_dict, writer=writer)
        # save model checkpoint
        if cfg["save_all"]:
            torch.save(model.state_dict(), os.path.join(save, 'epoch_{}'.format(epoch), 'model.pth'))

        if log_dict['val_dice_global'] > best_dice_global:
            torch.save(model.state_dict(), os.path.join(save, 'model.pth'))
            best_dice_global = log_dict['val_dice_global']
            dice_to_noise = log_dict['val_dice']

            print('New best global dice: %.4f, dice: %.4f' % (log_dict['val_dice_global'], log_dict['val_dice']))
        else:
            print('Current best global dice: %.4f, dice: %.4f' % (best_dice_global, dice_to_noise))
    # end 
    writer.close()
    with open(os.path.join(save, 'logs.csv'), 'a') as f:
        f.write(',,,,best global dice,%0.5f\n' % (best_dice_global))
    print('best global dice: ', best_dice_global)


def train_epoch(model, loader, optimizer, epoch, n_epochs, cfg, print_freq=1, writer=None):
 
    meters = MultiAverageMeter()
    # Model on train mode
    model.train()
    global iteration
    intersection = 0
    union = 0
    end = time.time()
    for batch_idx, (x, y) in enumerate(loader):
        x = to_var(x)
        y = to_var(y)

        pred_logit = model(x)
        y_one_hot = categorical_to_one_hot(y, dim=1, expand_dim=False)

        loss = soft_dice_loss(pred_logit, y_one_hot)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred_classes = pred_logit.argmax(1)
        intersection += ((pred_classes==1) * (y[:,0]==1)).sum().item()
        union += ((pred_classes==1).sum() + y[:,0].sum()).item()
        batch_size = y.size(0)

        iou = cal_batch_iou(pred_logit, y_one_hot)
        dice = cal_batch_dice(pred_logit, y_one_hot)

        writer.add_scalar('train_loss_logs', loss.item(), iteration)
 
        iteration += 1

        logs = [loss.item(), iou[1:].mean(), dice[1:].mean()]+ \
                            [iou[i].item() for i in range(len(iou))]+ \
                            [dice[i].item() for i in range(len(dice))]+ \
                            [time.time() - end]
        meters.update(logs, batch_size)   
        end = time.time()

        # print stats
        print_freq = 2 // meters.val[-1] + 1
        if batch_idx % print_freq == 0:
            res = '\t'.join([
                'Epoch: [%d/%d]' % (epoch + 1, n_epochs),
                'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
                'Time %.3f (%.3f)' % (meters.val[-1], meters.avg[-1]),
                'Loss %.4f (%.4f)' % (meters.val[0], meters.avg[0]),
                'IOU %.4f (%.4f)' % (meters.val[1], meters.avg[1]),
                'DICE %.4f (%.4f)' % (meters.val[2], meters.avg[2]),
            ])
            print(res)
    dice_global = 2. * intersection / union
    return meters.avg[:-1] + [dice_global]


def test_epoch(model, loader, print_freq=1, is_test=True):
    '''
    One test epoch
    '''
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
