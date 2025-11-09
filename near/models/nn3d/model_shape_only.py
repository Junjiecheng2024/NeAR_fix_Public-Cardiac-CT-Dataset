import torch
import torch.nn as nn
import torch.nn.functional as F

from near.models.nn3d.blocks import LatentCodeUpsample, ConvNormAct

DEFAULT = {
    "norm": lambda c: nn.GroupNorm(8, c),
    "activation": nn.LeakyReLU
}


class ImplicitDecoderShapeOnly(nn.Module):
    """
    Shape-only implicit decoder with improved MLP architecture.
    Based on the original ImplicitDecoder but removes appearance input
    and uses a deeper MLP with skip connection.
    """
    def __init__(self, latent_dimension, out_channels, norm, activation,
                 decoder_channels=[64, 48, 32, 16]):
        super().__init__()

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

        # in_ch = 3 (grid) + sum(decoder_channels)
        # For default decoder_channels=[64, 48, 32, 16]: 3 + 160 = 163
        in_ch = 3 + sum(decoder_channels)
        
        # Improved MLP with skip connection
        # fc1: in_ch -> 256 + ReLU
        self.fc1 = nn.Sequential(
            nn.Conv3d(in_ch, 256, kernel_size=1, padding=0),
            norm(256), activation()
        )
        
        # fc2: 256 -> 256 + ReLU
        self.fc2 = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=1, padding=0),
            norm(256), activation()
        )
        
        # skip: concat([h:256, input:in_ch]) -> 256+in_ch
        # fc3: (256+in_ch) -> 128 + ReLU
        self.fc3 = nn.Sequential(
            nn.Conv3d(256 + in_ch, 128, kernel_size=1, padding=0),
            norm(128), activation()
        )
        
        # fc4: 128 -> 64 + ReLU
        self.fc4 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=1, padding=0),
            norm(64), activation()
        )
        
        # output: 64 -> 1 (single class occupancy)
        self.output = nn.Conv3d(64, out_channels, kernel_size=1, padding=0)
        
        # Initialize output bias to favor negative class (background)
        # For extreme class imbalance (e.g., 0.86% positive), we want initial
        # predictions to be mostly negative to avoid the model being "confused"
        # Initial bias = log(p / (1-p)) where p is expected positive ratio
        # For p=0.01, bias = log(0.01/0.99) ≈ -4.6
        # This makes sigmoid(bias) ≈ 0.01, matching the data distribution
        with torch.no_grad():
            self.output.bias.fill_(-4.6)  # Biases initial predictions toward background

    def forward(self, x, grid):
        """
        Args:
            x: latent code (B, latent_dim)
            grid: sampling grid (B, D, H, W, 3)
        Returns:
            implicit_feature: concatenated features (B, in_ch, D, H, W)
            out: occupancy prediction (B, 1, D, H, W)
        """
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

        # Concatenate grid coordinates and decoder features (shape-only, no appearance)
        implicit_feature = torch.cat([grid.permute(0, 4, 1, 2, 3),
                                    implicit_feature_1,
                                    implicit_feature_2,
                                    implicit_feature_3,
                                    implicit_feature_4], dim=1)

        # Improved MLP with skip connection
        h1 = self.fc1(implicit_feature)
        h2 = self.fc2(h1)
        
        # Skip connection: concatenate h2 with input
        h_skip = torch.cat([h2, implicit_feature], dim=1)
        
        h3 = self.fc3(h_skip)
        h4 = self.fc4(h3)
        out = self.output(h4)

        return implicit_feature, out


class EmbeddingDecoderShapeOnly(nn.Module):
    """
    Shape-only embedding decoder for single-class NeAR refinement.
    """
    def __init__(self, latent_dimension=256, n_samples=998, decoder_channels=[64, 48, 32, 16]):
        super().__init__()

        self.latent_dimension = latent_dimension
        self.norm = DEFAULT["norm"]
        self.activation = DEFAULT["activation"]

        self.encoder = nn.Embedding(n_samples, latent_dimension)

        self.decoder = ImplicitDecoderShapeOnly(
            latent_dimension,
            out_channels=1,  # Single class output
            norm=self.norm,
            activation=self.activation,
            decoder_channels=decoder_channels
        )

    def forward(self, indices, grid):
        """
        Args:
            indices: sample indices (B,)
            grid: sampling grid (B, D, H, W, 3)
        Returns:
            out: occupancy prediction (B, 1, D, H, W)
            encoded: latent vectors (B, latent_dim)
        """
        encoded = self.encoder(indices)
        _, out = self.decoder(encoded, grid)

        return out, encoded
