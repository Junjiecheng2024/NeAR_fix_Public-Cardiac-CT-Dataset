import torch
import torch.nn as nn


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 norm=nn.BatchNorm3d, activation=nn.ReLU):
        if norm is None:
            def norm(_): return nn.Identity()
        padding = kernel_size // 2
        layers = [
            nn.Conv3d(in_channels, out_channels,
                      kernel_size=kernel_size, padding=padding),
            norm(out_channels), activation()
        ]
        super().__init__(*layers)


class LatentCodeUpsample(nn.Module):
    def __init__(self, in_channels, upsample_factor=2, channel_reduction=1,
                 norm=nn.BatchNorm3d, activation=nn.ReLU):
        super().__init__()
        if norm is None:
            def norm(_): return nn.Identity()
        out_channels = in_channels * (upsample_factor**3) // channel_reduction
        self.channel_reduction = channel_reduction
        self.upsample_factor = upsample_factor
        self.mlp = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0),
            norm(out_channels), activation())

    def forward(self, x):
        b, c, d, h, w = x.shape
        feature = self.mlp(x)
        o = feature.view(b, c // self.channel_reduction,
                         d * self.upsample_factor, h * self.upsample_factor,
                         w * self.upsample_factor)
        return o
