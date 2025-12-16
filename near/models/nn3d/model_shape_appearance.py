"""
Shape + Appearance model for NeAR v2.0 Tier2 training.
Adds CT appearance branch to improve fine structure learning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from near.models.nn3d.blocks import LatentCodeUpsample, ConvNormAct


DEFAULT = {
    "norm": lambda c: nn.GroupNorm(8, c),
    "activation": nn.LeakyReLU
}


class AppearanceEncoder(nn.Module):
    """
    Lightweight CNN encoder for CT appearance features.
    Extracts multi-scale features from CT volume.
    """
    def __init__(self, in_channels=1, base_channels=32, out_channels=64,
                 norm=None, activation=nn.LeakyReLU):
        super().__init__()
        
        if norm is None:
            norm = lambda c: nn.GroupNorm(min(8, c), c)
        
        # Multi-scale feature extraction
        # Level 1: Full resolution features
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1),
            norm(base_channels),
            activation(),
            nn.Conv3d(base_channels, base_channels, kernel_size=3, padding=1),
            norm(base_channels),
            activation()
        )
        
        # Level 2: 1/2 resolution
        self.down1 = nn.Conv3d(base_channels, base_channels, kernel_size=3, 
                               stride=2, padding=1)
        self.enc2 = nn.Sequential(
            nn.Conv3d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            norm(base_channels * 2),
            activation(),
            nn.Conv3d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            norm(base_channels * 2),
            activation()
        )
        
        # Level 3: 1/4 resolution  
        self.down2 = nn.Conv3d(base_channels * 2, base_channels * 2, kernel_size=3,
                               stride=2, padding=1)
        self.enc3 = nn.Sequential(
            nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            norm(base_channels * 4),
            activation(),
            nn.Conv3d(base_channels * 4, out_channels, kernel_size=3, padding=1),
            norm(out_channels),
            activation()
        )
        
        self.out_channels = out_channels
        # Multi-scale output channels: enc1 + enc2 + enc3
        self.multi_scale_channels = base_channels + base_channels * 2 + out_channels
        
    def forward(self, x):
        """
        Args:
            x: CT volume (B, 1, D, H, W)
        Returns:
            features: multi-scale feature dict for grid sampling
        """
        # Level 1
        f1 = self.enc1(x)  # (B, 32, D, H, W)
        
        # Level 2
        f2 = self.down1(f1)  # (B, 32, D/2, H/2, W/2)
        f2 = self.enc2(f2)   # (B, 64, D/2, H/2, W/2)
        
        # Level 3
        f3 = self.down2(f2)  # (B, 64, D/4, H/4, W/4)
        f3 = self.enc3(f3)   # (B, 64, D/4, H/4, W/4)
        
        return {"f1": f1, "f2": f2, "f3": f3}


class ImplicitDecoderShapeAppearance(nn.Module):
    """
    Shape + Appearance implicit decoder.
    Combines latent shape prior with CT appearance features.
    """
    def __init__(self, latent_dimension, out_channels=1, norm=None, activation=nn.LeakyReLU,
                 decoder_channels=[64, 48, 32, 16], appearance_channels=64):
        super().__init__()
        
        if norm is None:
            norm = DEFAULT["norm"]
        
        # Shape decoder (same as shape-only version)
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
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            ConvNormAct(decoder_channels[0], decoder_channels[1],
                        norm=norm, activation=activation),
            ConvNormAct(decoder_channels[1], decoder_channels[1],
                        norm=norm, activation=activation))

        self.decoder_3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            ConvNormAct(decoder_channels[1], decoder_channels[2],
                        norm=norm, activation=activation),
            ConvNormAct(decoder_channels[2], decoder_channels[2],
                        norm=norm, activation=activation))

        self.decoder_4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            ConvNormAct(decoder_channels[2], decoder_channels[3],
                        norm=norm, activation=activation),
            ConvNormAct(decoder_channels[3], decoder_channels[3],
                        norm=norm, activation=activation))

        # Appearance encoder
        self.appearance_encoder = AppearanceEncoder(
            in_channels=1,
            base_channels=32,
            out_channels=appearance_channels,
            norm=norm,
            activation=activation
        )
        
        # Feature dimensions
        # Shape: 3 (grid) + sum(decoder_channels) = 3 + 160 = 163
        # Appearance: 32 + 64 + 64 = 160 (multi-scale)
        shape_channels = 3 + sum(decoder_channels)
        app_channels = self.appearance_encoder.multi_scale_channels
        in_ch = shape_channels + app_channels  # 163 + 160 = 323
        
        # Improved MLP with skip connection (larger capacity for appearance)
        self.fc1 = nn.Sequential(
            nn.Conv3d(in_ch, 256, kernel_size=1, padding=0),
            norm(256), activation()
        )
        
        self.fc2 = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=1, padding=0),
            norm(256), activation()
        )
        
        # Skip connection
        self.fc3 = nn.Sequential(
            nn.Conv3d(256 + in_ch, 128, kernel_size=1, padding=0),
            norm(128), activation()
        )
        
        self.fc4 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=1, padding=0),
            norm(64), activation()
        )
        
        self.output = nn.Conv3d(64, out_channels, kernel_size=1, padding=0)
        
        # Initialize output bias (favor background)
        with torch.no_grad():
            self.output.bias.fill_(-4.6)

    def forward(self, latent, grid, appearance):
        """
        Args:
            latent: latent code (B, latent_dim)
            grid: sampling grid (B, D, H, W, 3), normalized to [-1, 1]
            appearance: CT volume (B, 1, D, H, W)
        Returns:
            implicit_feature: concatenated features
            out: occupancy prediction (B, 1, D, H, W)
        """
        # Shape branch
        x = latent.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        feature_map_1 = self.decoder_1(x)
        feature_map_2 = self.decoder_2(feature_map_1)
        feature_map_3 = self.decoder_3(feature_map_2)
        feature_map_4 = self.decoder_4(feature_map_3)

        # Sample shape features at grid locations
        implicit_feature_1 = F.grid_sample(feature_map_1, grid, mode="bilinear", align_corners=True)
        implicit_feature_2 = F.grid_sample(feature_map_2, grid, mode="bilinear", align_corners=True)
        implicit_feature_3 = F.grid_sample(feature_map_3, grid, mode="bilinear", align_corners=True)
        implicit_feature_4 = F.grid_sample(feature_map_4, grid, mode="bilinear", align_corners=True)

        # Appearance branch
        app_features = self.appearance_encoder(appearance)
        
        # Sample appearance features at grid locations
        app_f1 = F.grid_sample(app_features["f1"], grid, mode="bilinear", align_corners=True)
        app_f2 = F.grid_sample(app_features["f2"], grid, mode="bilinear", align_corners=True)
        app_f3 = F.grid_sample(app_features["f3"], grid, mode="bilinear", align_corners=True)

        # Concatenate all features
        implicit_feature = torch.cat([
            grid.permute(0, 4, 1, 2, 3),  # (B, 3, D, H, W)
            implicit_feature_1,
            implicit_feature_2,
            implicit_feature_3,
            implicit_feature_4,
            app_f1,
            app_f2,
            app_f3
        ], dim=1)

        # MLP with skip connection
        h1 = self.fc1(implicit_feature)
        h2 = self.fc2(h1)
        h_skip = torch.cat([h2, implicit_feature], dim=1)
        h3 = self.fc3(h_skip)
        h4 = self.fc4(h3)
        out = self.output(h4)

        return implicit_feature, out


class EmbeddingDecoderShapeAppearance(nn.Module):
    """
    Shape + Appearance embedding decoder for Tier2 NeAR.
    Uses per-sample learnable embeddings combined with CT appearance.
    """
    def __init__(self, latent_dimension=256, n_samples=998, 
                 decoder_channels=[64, 48, 32, 16], appearance_channels=64):
        super().__init__()

        self.latent_dimension = latent_dimension
        self.norm = DEFAULT["norm"]
        self.activation = DEFAULT["activation"]

        # Per-sample learnable embeddings
        self.encoder = nn.Embedding(n_samples, latent_dimension)

        # Shape + Appearance decoder
        self.decoder = ImplicitDecoderShapeAppearance(
            latent_dimension,
            out_channels=1,
            norm=self.norm,
            activation=self.activation,
            decoder_channels=decoder_channels,
            appearance_channels=appearance_channels
        )

    def forward(self, indices, grid, appearance):
        """
        Args:
            indices: sample indices (B,)
            grid: sampling grid (B, D, H, W, 3)
            appearance: CT volume (B, 1, D, H, W)
        Returns:
            out: occupancy prediction (B, 1, D, H, W)
            encoded: latent vectors (B, latent_dim)
        """
        encoded = self.encoder(indices)
        _, out = self.decoder(encoded, grid, appearance)

        return out, encoded
    
    def forward_with_latent(self, latent, grid, appearance):
        """
        Forward pass with explicit latent code (for inference with interpolated latents).
        
        Args:
            latent: latent code (B, latent_dim)
            grid: sampling grid (B, D, H, W, 3)
            appearance: CT volume (B, 1, D, H, W)
        Returns:
            out: occupancy prediction (B, 1, D, H, W)
        """
        _, out = self.decoder(latent, grid, appearance)
        return out


class EmbeddingDecoderShapeAppearanceWithContext(nn.Module):
    """
    Extended Shape + Appearance decoder that also takes context mask as input.
    The context mask (Myocardium + Aorta) provides spatial guidance for coronary prediction.
    """
    def __init__(self, latent_dimension=256, n_samples=998,
                 decoder_channels=[64, 48, 32, 16], appearance_channels=64,
                 use_context=True):
        super().__init__()
        
        self.use_context = use_context
        self.latent_dimension = latent_dimension
        self.norm = DEFAULT["norm"]
        self.activation = DEFAULT["activation"]
        
        # Per-sample learnable embeddings
        self.encoder = nn.Embedding(n_samples, latent_dimension)
        
        # If using context, add a small encoder for context mask
        if use_context:
            self.context_encoder = nn.Sequential(
                nn.Conv3d(1, 16, kernel_size=3, padding=1),
                self.norm(16),
                self.activation(),
                nn.Conv3d(16, 16, kernel_size=3, padding=1),
                self.norm(16),
                self.activation()
            )
            # Appearance encoder input: CT (1) + context features (will be merged)
            appearance_in_channels = 1 + 16
        else:
            self.context_encoder = None
            appearance_in_channels = 1
        
        # Modified appearance encoder that can take additional context
        self.appearance_encoder = AppearanceEncoder(
            in_channels=appearance_in_channels,
            base_channels=32,
            out_channels=appearance_channels,
            norm=self.norm,
            activation=self.activation
        )
        
        # Shape decoder (same structure)
        self.decoder_1 = nn.Sequential(
            LatentCodeUpsample(latent_dimension, upsample_factor=2, channel_reduction=2,
                               norm=self.norm, activation=self.activation),
            LatentCodeUpsample(latent_dimension // 2, upsample_factor=2, channel_reduction=2,
                               norm=self.norm, activation=self.activation),
            ConvNormAct(latent_dimension // 4, decoder_channels[0],
                        norm=self.norm, activation=self.activation))

        self.decoder_2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            ConvNormAct(decoder_channels[0], decoder_channels[1],
                        norm=self.norm, activation=self.activation),
            ConvNormAct(decoder_channels[1], decoder_channels[1],
                        norm=self.norm, activation=self.activation))

        self.decoder_3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            ConvNormAct(decoder_channels[1], decoder_channels[2],
                        norm=self.norm, activation=self.activation),
            ConvNormAct(decoder_channels[2], decoder_channels[2],
                        norm=self.norm, activation=self.activation))

        self.decoder_4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            ConvNormAct(decoder_channels[2], decoder_channels[3],
                        norm=self.norm, activation=self.activation),
            ConvNormAct(decoder_channels[3], decoder_channels[3],
                        norm=self.norm, activation=self.activation))
        
        # Feature dimensions
        shape_channels = 3 + sum(decoder_channels)  # 163
        app_channels = self.appearance_encoder.multi_scale_channels  # 160
        in_ch = shape_channels + app_channels
        
        # MLP
        self.fc1 = nn.Sequential(
            nn.Conv3d(in_ch, 256, kernel_size=1), self.norm(256), self.activation())
        self.fc2 = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=1), self.norm(256), self.activation())
        self.fc3 = nn.Sequential(
            nn.Conv3d(256 + in_ch, 128, kernel_size=1), self.norm(128), self.activation())
        self.fc4 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=1), self.norm(64), self.activation())
        self.output = nn.Conv3d(64, 1, kernel_size=1)
        
        with torch.no_grad():
            self.output.bias.fill_(-4.6)
    
    def forward(self, indices, grid, appearance, context=None):
        """
        Args:
            indices: sample indices (B,)
            grid: sampling grid (B, D, H, W, 3)
            appearance: CT volume (B, 1, D, H, W)
            context: context mask (B, 1, D, H, W) for Myo + Aorta, optional
        Returns:
            out: occupancy prediction (B, 1, D, H, W)
            encoded: latent vectors (B, latent_dim)
        """
        # Encode index to latent
        encoded = self.encoder(indices)
        
        # Process context if provided
        if self.use_context and context is not None:
            context_feat = self.context_encoder(context)
            app_input = torch.cat([appearance, context_feat], dim=1)
        else:
            app_input = appearance
        
        # Appearance features
        app_features = self.appearance_encoder(app_input)
        
        # Shape features
        x = encoded.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        f1 = self.decoder_1(x)
        f2 = self.decoder_2(f1)
        f3 = self.decoder_3(f2)
        f4 = self.decoder_4(f3)
        
        # Sample at grid locations
        sf1 = F.grid_sample(f1, grid, mode="bilinear", align_corners=True)
        sf2 = F.grid_sample(f2, grid, mode="bilinear", align_corners=True)
        sf3 = F.grid_sample(f3, grid, mode="bilinear", align_corners=True)
        sf4 = F.grid_sample(f4, grid, mode="bilinear", align_corners=True)
        
        af1 = F.grid_sample(app_features["f1"], grid, mode="bilinear", align_corners=True)
        af2 = F.grid_sample(app_features["f2"], grid, mode="bilinear", align_corners=True)
        af3 = F.grid_sample(app_features["f3"], grid, mode="bilinear", align_corners=True)
        
        # Concatenate
        implicit_feature = torch.cat([
            grid.permute(0, 4, 1, 2, 3),
            sf1, sf2, sf3, sf4,
            af1, af2, af3
        ], dim=1)
        
        # MLP
        h1 = self.fc1(implicit_feature)
        h2 = self.fc2(h1)
        h_skip = torch.cat([h2, implicit_feature], dim=1)
        h3 = self.fc3(h_skip)
        h4 = self.fc4(h3)
        out = self.output(h4)
        
        return out, encoded


class SobelGradient3D(nn.Module):
    """
    Computes 3D Sobel gradients for edge detection.
    Returns gradient magnitude and direction features.
    """
    def __init__(self):
        super().__init__()
        # 3D Sobel kernels
        sobel_x = torch.tensor([
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 16.0
        
        sobel_y = sobel_x.permute(0, 1, 2, 4, 3)  # Rotate for Y direction
        sobel_z = sobel_x.permute(0, 1, 4, 3, 2)  # Rotate for Z direction
        
        # Register as buffers (not learnable, but moves to GPU)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        self.register_buffer('sobel_z', sobel_z)
    
    def forward(self, x):
        """
        Args:
            x: CT volume (B, 1, D, H, W)
        Returns:
            gradient: (B, 4, D, H, W) - [gx, gy, gz, magnitude]
        """
        gx = F.conv3d(x, self.sobel_x, padding=1)
        gy = F.conv3d(x, self.sobel_y, padding=1)
        gz = F.conv3d(x, self.sobel_z, padding=1)
        
        # Gradient magnitude
        magnitude = torch.sqrt(gx**2 + gy**2 + gz**2 + 1e-8)
        
        return torch.cat([gx, gy, gz, magnitude], dim=1)


class FusionDecoderShapeAppearance(nn.Module):
    """
    NeAR v2.1 Fusion Model - Best of Both Worlds
    
    Combines:
    1. Multi-scale learned appearance features (from current Tier2)
    2. Raw CT values (from original HINTLab NeAR) 
    3. 3D Sobel gradients for edge detection (new)
    4. Context mask (Myocardium + Aorta) for spatial guidance
    5. Skip connection MLP with larger capacity
    
    This fusion approach provides both learned features AND 
    precise low-level boundary information for fine structures like coronary arteries.
    """
    def __init__(self, latent_dimension=256, n_samples=998,
                 decoder_channels=[64, 48, 32, 16], appearance_channels=64,
                 use_context=True, use_gradient=True, use_raw_ct=True):
        super().__init__()
        
        self.use_context = use_context
        self.use_gradient = use_gradient
        self.use_raw_ct = use_raw_ct
        self.latent_dimension = latent_dimension
        self.norm = DEFAULT["norm"]
        self.activation = DEFAULT["activation"]
        
        # Per-sample learnable embeddings
        self.encoder = nn.Embedding(n_samples, latent_dimension)
        
        # 3D Gradient extractor (non-learnable)
        if use_gradient:
            self.gradient_extractor = SobelGradient3D()
        
        # Context encoder
        if use_context:
            self.context_encoder = nn.Sequential(
                nn.Conv3d(1, 16, kernel_size=3, padding=1),
                self.norm(16),
                self.activation(),
                nn.Conv3d(16, 16, kernel_size=3, padding=1),
                self.norm(16),
                self.activation()
            )
            appearance_in_channels = 1 + 16
        else:
            self.context_encoder = None
            appearance_in_channels = 1
        
        # Multi-scale appearance encoder
        self.appearance_encoder = AppearanceEncoder(
            in_channels=appearance_in_channels,
            base_channels=32,
            out_channels=appearance_channels,
            norm=self.norm,
            activation=self.activation
        )
        
        # Shape decoder
        self.decoder_1 = nn.Sequential(
            LatentCodeUpsample(latent_dimension, upsample_factor=2, channel_reduction=2,
                               norm=self.norm, activation=self.activation),
            LatentCodeUpsample(latent_dimension // 2, upsample_factor=2, channel_reduction=2,
                               norm=self.norm, activation=self.activation),
            ConvNormAct(latent_dimension // 4, decoder_channels[0],
                        norm=self.norm, activation=self.activation))

        self.decoder_2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            ConvNormAct(decoder_channels[0], decoder_channels[1],
                        norm=self.norm, activation=self.activation),
            ConvNormAct(decoder_channels[1], decoder_channels[1],
                        norm=self.norm, activation=self.activation))

        self.decoder_3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            ConvNormAct(decoder_channels[1], decoder_channels[2],
                        norm=self.norm, activation=self.activation),
            ConvNormAct(decoder_channels[2], decoder_channels[2],
                        norm=self.norm, activation=self.activation))

        self.decoder_4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            ConvNormAct(decoder_channels[2], decoder_channels[3],
                        norm=self.norm, activation=self.activation),
            ConvNormAct(decoder_channels[3], decoder_channels[3],
                        norm=self.norm, activation=self.activation))
        
        # Calculate feature dimensions
        # Grid: 3 channels
        # Shape: sum(decoder_channels) = 160
        # Multi-scale appearance: 32 + 64 + 64 = 160
        # Raw CT: 1 (optional)
        # Gradient: 4 (gx, gy, gz, magnitude) (optional)
        shape_channels = 3 + sum(decoder_channels)  # 163
        app_channels = self.appearance_encoder.multi_scale_channels  # 160
        
        extra_channels = 0
        if use_raw_ct:
            extra_channels += 1
        if use_gradient:
            extra_channels += 4
        
        in_ch = shape_channels + app_channels + extra_channels
        
        # Larger MLP with residual-style skip connection
        self.fc1 = nn.Sequential(
            nn.Conv3d(in_ch, 256, kernel_size=1), self.norm(256), self.activation())
        self.fc2 = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=1), self.norm(256), self.activation())
        # Skip connection: concat fc2 output with original features
        self.fc3 = nn.Sequential(
            nn.Conv3d(256 + in_ch, 128, kernel_size=1), self.norm(128), self.activation())
        self.fc4 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=1), self.norm(64), self.activation())
        self.output = nn.Conv3d(64, 1, kernel_size=1)
        
        # Initialize output bias to favor background (helps with small structures)
        with torch.no_grad():
            self.output.bias.fill_(-4.6)
        
        # Print model config
        print(f"[FusionDecoderShapeAppearance] Initialized:")
        print(f"  - Latent dim: {latent_dimension}, N samples: {n_samples}")
        print(f"  - Use context: {use_context}, Use gradient: {use_gradient}, Use raw CT: {use_raw_ct}")
        print(f"  - Total feature channels: {in_ch}")
    
    def forward(self, indices, grid, appearance, context=None):
        """
        Args:
            indices: sample indices (B,)
            grid: sampling grid (B, D, H, W, 3), normalized to [-1, 1]
            appearance: CT volume (B, 1, D, H, W), normalized to [0, 1]
            context: context mask (B, 1, D, H, W) for Myo + Aorta, optional
        Returns:
            out: occupancy prediction logits (B, 1, D, H, W)
            encoded: latent vectors (B, latent_dim)
        """
        batch_size = indices.shape[0]
        
        # 1. Encode sample index to latent code
        encoded = self.encoder(indices)
        
        # 2. Process context if provided
        if self.use_context and context is not None:
            context_feat = self.context_encoder(context)
            app_input = torch.cat([appearance, context_feat], dim=1)
        else:
            app_input = appearance
        
        # 3. Extract multi-scale appearance features
        app_features = self.appearance_encoder(app_input)
        
        # 4. Shape decoder: latent → multi-scale shape features
        x = encoded.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        f1 = self.decoder_1(x)
        f2 = self.decoder_2(f1)
        f3 = self.decoder_3(f2)
        f4 = self.decoder_4(f3)
        
        # 5. Sample shape features at grid locations
        sf1 = F.grid_sample(f1, grid, mode="bilinear", align_corners=True)
        sf2 = F.grid_sample(f2, grid, mode="bilinear", align_corners=True)
        sf3 = F.grid_sample(f3, grid, mode="bilinear", align_corners=True)
        sf4 = F.grid_sample(f4, grid, mode="bilinear", align_corners=True)
        
        # 6. Sample appearance features at grid locations
        af1 = F.grid_sample(app_features["f1"], grid, mode="bilinear", align_corners=True)
        af2 = F.grid_sample(app_features["f2"], grid, mode="bilinear", align_corners=True)
        af3 = F.grid_sample(app_features["f3"], grid, mode="bilinear", align_corners=True)
        
        # 7. Build feature list for concatenation
        features_to_concat = [
            grid.permute(0, 4, 1, 2, 3),  # (B, 3, D, H, W) - positional encoding
            sf1, sf2, sf3, sf4,            # Shape features
            af1, af2, af3                   # Multi-scale appearance features
        ]
        
        # 8. Add raw CT values (like original HINTLab NeAR)
        if self.use_raw_ct:
            raw_ct_sampled = F.grid_sample(appearance, grid, mode="bilinear", align_corners=True)
            features_to_concat.append(raw_ct_sampled)
        
        # 9. Add gradient features for edge detection
        if self.use_gradient:
            gradient = self.gradient_extractor(appearance)
            gradient_sampled = F.grid_sample(gradient, grid, mode="bilinear", align_corners=True)
            features_to_concat.append(gradient_sampled)
        
        # 10. Concatenate all features
        implicit_feature = torch.cat(features_to_concat, dim=1)
        
        # 11. MLP with skip connection
        h1 = self.fc1(implicit_feature)
        h2 = self.fc2(h1)
        h_skip = torch.cat([h2, implicit_feature], dim=1)  # Residual-style skip
        h3 = self.fc3(h_skip)
        h4 = self.fc4(h3)
        out = self.output(h4)
        
        return out, encoded
    
    def forward_with_latent(self, latent, grid, appearance, context=None):
        """
        Forward pass with explicit latent code (for inference with mean/interpolated latents).
        """
        # Process context if provided
        if self.use_context and context is not None:
            context_feat = self.context_encoder(context)
            app_input = torch.cat([appearance, context_feat], dim=1)
        else:
            app_input = appearance
        
        # Extract appearance features
        app_features = self.appearance_encoder(app_input)
        
        # Shape decoder
        x = latent.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        f1 = self.decoder_1(x)
        f2 = self.decoder_2(f1)
        f3 = self.decoder_3(f2)
        f4 = self.decoder_4(f3)
        
        # Sample at grid locations
        sf1 = F.grid_sample(f1, grid, mode="bilinear", align_corners=True)
        sf2 = F.grid_sample(f2, grid, mode="bilinear", align_corners=True)
        sf3 = F.grid_sample(f3, grid, mode="bilinear", align_corners=True)
        sf4 = F.grid_sample(f4, grid, mode="bilinear", align_corners=True)
        
        af1 = F.grid_sample(app_features["f1"], grid, mode="bilinear", align_corners=True)
        af2 = F.grid_sample(app_features["f2"], grid, mode="bilinear", align_corners=True)
        af3 = F.grid_sample(app_features["f3"], grid, mode="bilinear", align_corners=True)
        
        features_to_concat = [
            grid.permute(0, 4, 1, 2, 3),
            sf1, sf2, sf3, sf4,
            af1, af2, af3
        ]
        
        if self.use_raw_ct:
            raw_ct_sampled = F.grid_sample(appearance, grid, mode="bilinear", align_corners=True)
            features_to_concat.append(raw_ct_sampled)
        
        if self.use_gradient:
            gradient = self.gradient_extractor(appearance)
            gradient_sampled = F.grid_sample(gradient, grid, mode="bilinear", align_corners=True)
            features_to_concat.append(gradient_sampled)
        
        implicit_feature = torch.cat(features_to_concat, dim=1)
        
        h1 = self.fc1(implicit_feature)
        h2 = self.fc2(h1)
        h_skip = torch.cat([h2, implicit_feature], dim=1)
        h3 = self.fc3(h_skip)
        h4 = self.fc4(h3)
        out = self.output(h4)
        
        return out
