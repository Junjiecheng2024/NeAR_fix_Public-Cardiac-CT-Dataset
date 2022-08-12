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
                 label_interpolation_mode="bilinear"):
        self.grid_sampler = UniformGridSampler(resolution)
        self.grid_noise = grid_noise
        self.uniform_grid_noise = uniform_grid_noise
        self.label_interpolation_mode = label_interpolation_mode

    def __call__(self, volumes):
        batch_size = volumes.shape[0]
        volumes = to_var(volumes)
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
