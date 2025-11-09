import torch
import torch.nn as nn
import torch.nn.functional as F

from near.utils.misc import to_var

IDENTITY_THETA = torch.tensor([1, 0, 0, 0,
                               0, 1, 0, 0,
                               0, 0, 1, 0],
                              dtype=torch.float)


def grid_affine_transform(grid, theta_diff):

    batch_size, channels = theta_diff.shape
    assert channels == 12
    theta = (theta_diff +
             IDENTITY_THETA.to(theta_diff.device)
             ).view(batch_size, 3, 4)
    w = theta[:, :, :-1]  # Bx3x3
    t = theta[:, :, -1]  # Bx3
    transformed = torch.einsum("bmn,bdhwn->bdhwm", w, grid) +\
        t.view(batch_size, 1, 1, 1, 3)
    return transformed


def generate_meshgrid_3d(resolution):
    d = torch.linspace(-1, 1, resolution)
    meshx, meshy, meshz = torch.meshgrid((d, d, d))
    grid = torch.stack((meshz, meshy, meshx), -1)
    return grid


class UniformGridSampler:
    def __init__(self, resolution):
        self.grid_cache = dict()
        self.resolution = resolution

    def generate_batch_grid(self, batch_size, resolution=None):
        resolution = resolution or self.resolution
        if resolution in self.grid_cache:
            grid = self.grid_cache[resolution]
        else:
            grid = generate_meshgrid_3d(resolution).unsqueeze(0)
        batch_grid = grid.repeat_interleave(batch_size, dim=0)
        return batch_grid


class AffineGridSampler(nn.Module):

    def __init__(self, resolution, trainable=False):
        super().__init__()

        self.resolution = resolution

        if trainable:
            self.identity_theta = nn.Parameter(IDENTITY_THETA)
        else:
            self.register_buffer("identity_theta", IDENTITY_THETA)

    def forward(self, theta_diff, resolution=None):

        resolution = resolution or self.resolution
        batch_size, channels = theta_diff.shape
        assert channels == 12
        theta = (theta_diff + self.identity_theta).view(batch_size, 3, 4)
        grid = F.affine_grid(theta,
                             size=(batch_size, 1) + (resolution, ) * 3,
                             align_corners=True)
        return grid

    def generate_batch_grid(self, batch_size, resolution=None):
        theta = to_var(torch.zeros(batch_size, 12))
        return self.forward(theta, resolution)


class GatherGridsFromVolumes:
    def __init__(self,
                 resolution=32,
                 grid_noise=None,
                 uniform_grid_noise=False,
                 label_interpolation_mode="bilinear",
                 boundary_bias_ratio=0.0,
                 boundary_dilation_radius=2):
        self.grid_sampler = UniformGridSampler(resolution)
        self.grid_noise = grid_noise
        self.uniform_grid_noise = uniform_grid_noise
        self.label_interpolation_mode = label_interpolation_mode
        self.boundary_bias_ratio = boundary_bias_ratio
        self.boundary_dilation_radius = boundary_dilation_radius
        
        if boundary_bias_ratio > 0:
            print(f"GatherGridsFromVolumes: Using boundary-biased sampling "
                  f"({boundary_bias_ratio*100:.0f}% near boundaries, "
                  f"dilation={boundary_dilation_radius})")

    def get_boundary_mask(self, volume):
        """
        Extract boundary region from volume.
        
        Args:
            volume: (B, 1, D, H, W) binary volume
        Returns:
            boundary_mask: (B, 1, D, H, W) binary boundary mask
        """
        from scipy.ndimage import binary_dilation
        import numpy as np
        
        batch_size = volume.shape[0]
        boundaries = []
        
        for i in range(batch_size):
            vol = volume[i, 0].cpu().numpy() > 0.5
            # Dilate
            dilated = binary_dilation(vol, iterations=self.boundary_dilation_radius)
            # Boundary = dilated XOR original
            boundary = np.logical_xor(dilated, vol).astype(np.float32)
            boundaries.append(boundary)
        
        boundary_volume = torch.from_numpy(np.stack(boundaries)).unsqueeze(1)
        return boundary_volume.to(volume.device)

    def sample_biased_grid(self, volumes, resolution):
        """
        Sample a grid with bias towards boundaries.
        
        Args:
            volumes: (B, 1, D, H, W) input volumes
            resolution: number of samples (will be resolution³)
        Returns:
            grids: (B, resolution, resolution, resolution, 3)
        """
        batch_size = volumes.shape[0]
        device = volumes.device
        
        # Get boundary mask
        boundary_mask = self.get_boundary_mask(volumes)
        
        # Calculate number of boundary and uniform samples
        total_samples = resolution ** 3
        n_boundary = int(total_samples * self.boundary_bias_ratio)
        n_uniform = total_samples - n_boundary
        
        # Generate uniform base grid
        base_grid = generate_meshgrid_3d(resolution).to(device)  # (D, H, W, 3)
        
        # For each sample in batch, create biased grid
        grids = []
        for i in range(batch_size):
            vol = volumes[i:i+1]  # (1, 1, D, H, W)
            boundary = boundary_mask[i:i+1]  # (1, 1, D, H, W)
            
            # Find boundary voxel coordinates in volume space
            boundary_coords = torch.nonzero(boundary[0, 0] > 0.5)  # (N, 3)
            
            if len(boundary_coords) > 0 and n_boundary > 0:
                # Sample random boundary points
                if len(boundary_coords) < n_boundary:
                    # If not enough boundary points, sample with replacement
                    boundary_indices = torch.randint(0, len(boundary_coords), 
                                                    (n_boundary,), device=device)
                else:
                    # Sample without replacement
                    boundary_indices = torch.randperm(len(boundary_coords), device=device)[:n_boundary]
                
                sampled_boundary_coords = boundary_coords[boundary_indices]  # (n_boundary, 3)
                
                # Convert to normalized coordinates [-1, 1]
                vol_size = torch.tensor(boundary.shape[2:], device=device).float()
                boundary_grid_coords = (sampled_boundary_coords.float() / (vol_size - 1)) * 2 - 1
                
                # Uniform samples (flatten base grid and sample randomly)
                uniform_flat = base_grid.reshape(-1, 3)
                uniform_indices = torch.randperm(uniform_flat.shape[0], device=device)[:n_uniform]
                uniform_grid_coords = uniform_flat[uniform_indices]
                
                # Combine boundary and uniform samples
                combined_coords = torch.cat([boundary_grid_coords, uniform_grid_coords], dim=0)
                
                # Shuffle to mix boundary and uniform samples
                shuffle_indices = torch.randperm(combined_coords.shape[0], device=device)
                combined_coords = combined_coords[shuffle_indices]
                
                # Reshape to grid format
                grid = combined_coords.reshape(resolution, resolution, resolution, 3)
            else:
                # No boundary points or no biased sampling, use uniform grid
                grid = base_grid
            
            grids.append(grid)
        
        grids = torch.stack(grids, dim=0)  # (B, D, H, W, 3)
        return grids

    def __call__(self, volumes):
        batch_size = volumes.shape[0]
        volumes = to_var(volumes)
        
        # Use biased sampling if enabled, otherwise uniform
        if self.boundary_bias_ratio > 0:
            grids = self.sample_biased_grid(volumes, self.grid_sampler.resolution)
            grids = to_var(grids)
        else:
            grids = to_var(self.grid_sampler.generate_batch_grid(batch_size))
        
        if self.grid_noise is not None:
            if self.uniform_grid_noise:
                grids += to_var(torch.randn(batch_size,
                                            1, 1, 1, 1)) * self.grid_noise
            else:
                grids += torch.randn_like(grids) * self.grid_noise
        
        labels = F.grid_sample(volumes,
                               grids,
                               mode=self.label_interpolation_mode,
                               align_corners=True)
        return volumes, grids, labels
