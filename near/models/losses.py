import torch
import torch.nn as nn
import torch.nn.functional as F


def batch_dice_score(y_pred, y_true):
    batch_size = y_pred.shape[0]
    assert y_true.shape[0] == batch_size
    smooth = 1e-6
    y_pred_ = y_pred.reshape(batch_size, -1)
    y_true_ = y_true.reshape(batch_size, -1)
    ret = (2 * (y_true_ * y_pred_).sum(-1) + smooth) / (y_true_.sum(-1) +
                                                        y_pred_.sum(-1) + smooth)
    return ret


def batch_binary_cross_entropy_with_logits(y_pred, y_true):
    batch_size = y_pred.shape[0]
    assert y_true.shape[0] == batch_size
    ret = F.binary_cross_entropy_with_logits(
        y_pred, y_true, reduction="none").reshape(batch_size, -1).mean(-1)
    return ret


def dice_score(y_pred, y_true):
    '''a.k.a. Dice Similarity Coefficient (DSC)'''
    smooth = 1e-6
    ret = (2 * (y_true * y_pred).sum() + smooth) / (y_true.sum() +
                                                    y_pred.sum() + smooth)
    return ret


def kl_divergence(mu, logvar):
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    return KLD


def l2_penalty(tensor):
    print("Keep for backward compatible only. Use `latent_l2_penalty` instead.")
    return latent_l2_penalty(tensor)


def latent_l2_penalty(tensor, reduce=True):
    batch_size = tensor.shape[0]
    l2 = tensor.reshape(batch_size, -1).norm(2, dim=-1)
    if reduce:
        return l2.mean()
    return l2


def max_deformation_penalty(tensor):
    batch_size = tensor.shape[0]
    maxd = tensor.reshape(batch_size, -1).abs().max(-1)[0].mean()
    return maxd


def avg_deformation_penalty(tensor):
    avgd = tensor.abs().mean()
    return avgd


def border_penalty(tensor):
    return (tensor.abs().max()-1).relu().mean()


class LaplacianLoss3d(nn.Module):

    def __init__(self, norm_order=2):
        super().__init__()
        diff_kernel = torch.zeros(3, 1, 3, 3, 3)
        diff_kernel[0, 0, :, 1, 1] = torch.tensor([1., -2., 1.])
        diff_kernel[1, 0, 1, :, 1] = torch.tensor([1., -2., 1.])
        diff_kernel[2, 0, 1, 1, :] = torch.tensor([1., -2., 1.])
        self.register_buffer("diff_kernel", diff_kernel)
        self.norm_order = norm_order

    def forward(self, inputs):
        input_channels = inputs.shape[1]  # BxCxDxHxW
        kernel = self.diff_kernel.repeat_interleave(input_channels, dim=0)

        padded = F.pad(inputs, (1, 1, 1, 1, 1, 1), mode="replicate")

        diff = F.conv3d(padded, kernel, groups=input_channels)

        norm = diff.norm(self.norm_order, dim=(1))  # Bx(D)x(H)x(W)
        return norm.mean()


class EikonalLoss3d(nn.Module):

    def __init__(self, norm_order=2):
        '''There is kind of border artifact.
        We could use reflect padding at right corner to reduce it, 
        but PyTorch do not support reflect padding 3D.
        '''
        super().__init__()
        diff_kernel = torch.zeros(3, 1, 2, 2, 2)
        diff_kernel[0, 0, :, 0, 0] = torch.tensor([1., -1.])
        diff_kernel[1, 0, 0, :, 0] = torch.tensor([1., -1.])
        diff_kernel[2, 0, 0, 0, :] = torch.tensor([1., -1.])
        self.register_buffer("diff_kernel", diff_kernel)
        self.norm_order = norm_order

    def forward(self, inputs):
        _, input_channels, *dhw = inputs.shape
        scale = self.diff_kernel.new(dhw).reshape(3, 1, 1, 1, 1)
        kernel = (self.diff_kernel *
                  scale).repeat_interleave(input_channels, dim=0)

        diff = F.conv3d(inputs, kernel, groups=input_channels)

        norm = diff.norm(self.norm_order, dim=(1))
        return (norm-1).abs().mean()


def implicit_sdf_loss(y_pred, y_true):
    return ((1-2*y_true)*y_pred).relu().mean()
