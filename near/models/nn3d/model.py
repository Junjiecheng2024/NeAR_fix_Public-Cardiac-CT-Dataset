import torch
import torch.nn as nn
import torch.nn.functional as F

from near.models.nn3d.blocks import LatentCodeUpsample, ConvNormAct

DEFAULT = {
    "norm": lambda c: nn.GroupNorm(8, c),
    "activation": nn.LeakyReLU
}


class ImplicitDecoder(nn.Module):
    def __init__(self, latent_dimension, out_channels, norm, activation,
                 decoder_channels=[64, 48, 32, 16], appearance=True):
        super().__init__()

        self.appearance = appearance

        self.decoder_1 = nn.Sequential(
            LatentCodeUpsample(latent_dimension,
                               upsample_factor=2,
                               channel_reduction=2,
                               norm=None if norm == nn.InstanceNorm3d else norm,
                               activation=activation),
            LatentCodeUpsample(latent_dimension // 2,
                               upsample_factor=2,
                               channel_reduction=2,
                               norm=norm,
                               activation=activation),
            ConvNormAct(latent_dimension // 4, decoder_channels[0],
                        norm=norm,
                        activation=activation))

        self.decoder_2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear',
                        align_corners=True),
            ConvNormAct(decoder_channels[0], decoder_channels[1],
                        norm=norm,
                        activation=activation),
            ConvNormAct(decoder_channels[1], decoder_channels[1],
                        norm=norm,
                        activation=activation))

        self.decoder_3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear',
                        align_corners=True),
            ConvNormAct(decoder_channels[1], decoder_channels[2],
                        norm=norm,
                        activation=activation),
            ConvNormAct(decoder_channels[2], decoder_channels[2],
                        norm=norm,
                        activation=activation))

        self.decoder_4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear',
                        align_corners=True),
            ConvNormAct(decoder_channels[2], decoder_channels[3],
                        norm=norm,
                        activation=activation),
            ConvNormAct(decoder_channels[3], decoder_channels[3],
                        norm=norm,
                        activation=activation))

        
        in_channels = 3 + sum(decoder_channels) + 1 if appearance else 3 + sum(decoder_channels)
        self.implicit_mlp = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=1, padding=0),
            norm(64), activation(),
            nn.Conv3d(64, 32, kernel_size=1, padding=0),
            norm(32), activation(),
            nn.Conv3d(32, out_channels, kernel_size=1, padding=0))

    def forward(self, x, grid, appearance):
        x = x.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        feature_map_1 = self.decoder_1(x)
        feature_map_2 = self.decoder_2(feature_map_1)
        feature_map_3 = self.decoder_3(feature_map_2)
        feature_map_4 = self.decoder_4(feature_map_3)

        implicit_feature_1 = F.grid_sample(feature_map_1,
                                           grid,
                                           mode="bilinear",
                                           align_corners=True)
        implicit_feature_2 = F.grid_sample(feature_map_2,
                                           grid,
                                           mode="bilinear",
                                           align_corners=True)
        implicit_feature_3 = F.grid_sample(feature_map_3,
                                           grid,
                                           mode="bilinear",
                                           align_corners=True)
        implicit_feature_4 = F.grid_sample(feature_map_4,
                                           grid,
                                           mode="bilinear",
                                           align_corners=True)

        if self.appearance:
            implicit_feature = torch.cat([grid.permute(0, 4, 1, 2, 3),
                                        implicit_feature_1,
                                        implicit_feature_2,
                                        implicit_feature_3,
                                        implicit_feature_4,
                                        appearance], dim=1)
        else:
            implicit_feature = torch.cat([grid.permute(0, 4, 1, 2, 3),
                                        implicit_feature_1,
                                        implicit_feature_2,
                                        implicit_feature_3,
                                        implicit_feature_4], dim=1)

        # shape mlp
        out = self.implicit_mlp(implicit_feature)

        return implicit_feature, out


class EmbeddingDecoder(nn.Module):
    def __init__(self, latent_dimension=256, n_samples=120, appearance=True):

        super().__init__()

        self.appearance = appearance

        self.latent_dimension = latent_dimension
        self.norm = DEFAULT["norm"]
        self.activation = DEFAULT["activation"]

        self.encoder = nn.Embedding(n_samples, latent_dimension)

        self.decoder = ImplicitDecoder(latent_dimension,
                                        out_channels=1,
                                        norm=self.norm,
                                        activation=self.activation,
                                        appearance=appearance)

    def forward(self, indices, grid, appearance):
        encoded = self.encoder(indices)
        _, out = self.decoder(encoded, grid, appearance)

        return out, encoded
