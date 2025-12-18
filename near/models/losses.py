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


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in binary segmentation.
    
    Args:
        alpha: Weighting factor for positive class (default: 0.25)
        gamma: Focusing parameter (default: 4.0)
        reduction: 'mean', 'sum', or 'none'
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


class BoundaryDiceLoss(nn.Module):
    """
    Dice loss computed only on boundary region.
    Helps model focus on fine boundary details.
    
    Args:
        boundary_width: Width of boundary region in voxels (default: 2)
        smooth: Smoothing factor to avoid division by zero (default: 1e-5)
    """
    def __init__(self, boundary_width=2, smooth=1e-5):
        super().__init__()
        self.boundary_width = boundary_width
        self.smooth = smooth
    
    def forward(self, pred_prob, target):
        """
        Args:
            pred_prob: predicted probability (B, 1, D, H, W)
            target: ground truth binary mask (B, 1, D, H, W)
        """
        # Compute boundary mask via dilation - erosion
        # Use max_pool for dilation, -max_pool(-x) for erosion
        kernel_size = 2 * self.boundary_width + 1
        padding = self.boundary_width
        
        dilated = F.max_pool3d(target, kernel_size, stride=1, padding=padding)
        eroded = -F.max_pool3d(-target, kernel_size, stride=1, padding=padding)
        boundary = (dilated - eroded).clamp(0, 1)
        
        # Compute Dice on boundary region only
        pred_boundary = pred_prob * boundary
        target_boundary = target * boundary
        
        intersection = (pred_boundary * target_boundary).sum()
        union = pred_boundary.sum() + target_boundary.sum()
        
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice


class TverskyLoss(nn.Module):
    """
    Tversky Loss - generalizes Dice loss with adjustable alpha/beta.
    Better for highly imbalanced segmentation (like coronary arteries).
    
    - alpha > beta: emphasizes recall (reduces false negatives / missed detections)
    - alpha < beta: emphasizes precision (reduces false positives)
    
    For small structures like coronary: use alpha=0.2, beta=0.8 to strongly prioritize recall.
    
    Args:
        alpha: weight for false positives (default: 0.2)
        beta: weight for false negatives (default: 0.8)
        smooth: smoothing factor (default: 1e-5)
    """
    def __init__(self, alpha=0.2, beta=0.8, smooth=1e-5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, pred_prob, target):
        """
        Args:
            pred_prob: predicted probability (B, 1, D, H, W)
            target: ground truth binary mask (B, 1, D, H, W)
        """
        pred_flat = pred_prob.reshape(-1)
        target_flat = target.reshape(-1)
        
        # True positives, false positives, false negatives
        tp = (pred_flat * target_flat).sum()
        fp = (pred_flat * (1 - target_flat)).sum()
        fn = ((1 - pred_flat) * target_flat).sum()
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky


class TopKLoss(nn.Module):
    """
    TopK Loss - focuses on the hardest K% of voxels.
    Helps with class imbalance by ignoring easy negatives.
    
    Args:
        k: percentage of hardest voxels to use (default: 0.1 = 10%)
    """
    def __init__(self, k=0.1):
        super().__init__()
        self.k = k
    
    def forward(self, pred_logit, target):
        """
        Args:
            pred_logit: raw logits (B, 1, D, H, W)
            target: ground truth binary mask (B, 1, D, H, W)
        """
        # Compute per-voxel BCE loss
        bce = F.binary_cross_entropy_with_logits(pred_logit, target, reduction='none')
        
        # Flatten and sort
        bce_flat = bce.reshape(-1)
        n_voxels = bce_flat.shape[0]
        k_voxels = max(int(n_voxels * self.k), 1)
        
        # Take top-k (hardest) losses
        topk_loss, _ = torch.topk(bce_flat, k_voxels)
        
        return topk_loss.mean()

