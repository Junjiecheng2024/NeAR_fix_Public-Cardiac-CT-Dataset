import json
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import torch

from surface_distance import compute_surface_distances, compute_surface_dice_at_tolerance

USE_GPU = True

if USE_GPU and torch.cuda.is_available():

    def to_device(x, gpu=None):
        x = x.cuda(gpu)
        return x

else:

    def to_device(x, gpu=None):
        return x


if USE_GPU and torch.cuda.is_available():

    def to_var(x, requires_grad=False, gpu=None):
        x = x.cuda(gpu)
        return x.requires_grad_(requires_grad)

else:

    def to_var(x, requires_grad=False, gpu=None):
        return x.requires_grad_(requires_grad)


def write_json(ctx, path, verbose=False):
    with open(path, "w") as f:
        json_string = json.dumps(ctx, indent=4)
        f.write(json_string)
    if verbose:
        print(json_string)


class Metrics:

    def __init__(self, *keys):
        self.keys = keys
        self.metrics = self.reset_dict()

    def reset_dict(self):
        ret = OrderedDict()
        for key in self.keys:
            ret[key] = []
        return ret

    def update(self, key, val):
        self.metrics[key].append(val)

    def ordered_update(self, *vals):
        for key, val in zip(self.keys, vals):
            self.update(key, val)

    def ordered_mean(self):
        ret = []
        for key in self.metrics:
            ret.append(np.mean(self.metrics[key]))
        return ret

    def log_latest(self, header="", verbose=True):
        ret = header
        for k, v in self.metrics.items():
            ret += f"\t[{k}] {v[-1]:.4f}"
        if verbose:
            print(ret)
        return ret

    def find_best(self, key, lower_is_better=True):
        if len(self.metrics[key]) == 0:
            return (np.inf, {}) if lower_is_better else (-np.inf, {})
        best_index, best = min(
            enumerate(self.metrics[key]), key=lambda t: t[1] if lower_is_better else -t[1])
        best_dict = dict(best_index=best_index)
        for key in self.keys:
            best_dict[key] = self.metrics[key][best_index]
        return best, best_dict

    def save(self, path):
        write_json(self.metrics, path)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            parsed = json.loads(f.read())
        ret = cls(*parsed.keys())
        for key in ret.keys:
            ret.metrics[key] = parsed[key]
        return ret

    def plot(self, key, marker='-'):
        plt.plot(self.metrics[key], marker)


def surface_dice(y_pred, y_true):
    '''a.k.a. Normalized Surface Distance (NSD)'''
    mask_gt, mask_pred = y_true, y_pred
    surface_distances = compute_surface_distances(
        mask_gt, mask_pred, spacing_mm=(1, 1, 1))
    ret = compute_surface_dice_at_tolerance(surface_distances, 1)
    return ret


def cal_surface_dice(y_pred, y_true):

    sdice_ret = []
    for i in range(y_pred.shape[0]):
        pred = y_pred[i].squeeze(0)
        true = y_true[i].squeeze(0)
        sdice_ret.append(surface_dice(pred, true).item())

    return np.mean(sdice_ret)


def load_model_on_multi_gpu(model, model_path, kwargs=None):

    if kwargs is not None:
        state_dict = torch.load(model_path, **kwargs)
    else:
        state_dict = torch.load(model_path)
    
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    return model


def load_model_on_single_gpu(model, model_path, kwargs=None):
    if kwargs is not None:
        state_dict = torch.load(model_path, **kwargs)
    else:
        state_dict = torch.load(model_path)
        
    model.load_state_dict(state_dict)
    return model
