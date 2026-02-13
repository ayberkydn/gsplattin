import torch
import torch.nn.functional as F
from gsplat import rasterization
from torch import nn

from .camera import CameraRanges


class GaussianSplat(nn.Module):
    """Gaussian splat with learnable parameters and batched rendering."""

    def __init__(self, num_gaussians: int) -> None:
        super().__init__()

        # Initialize gaussians around the origin.
        means = torch.randn(num_gaussians, 3, device="cuda") * 0.1
        self.means = nn.Parameter(means)

        # Quaternion rotation (w, x, y, z). Initialized to identity.
        quats = torch.zeros(num_gaussians, 4, device="cuda")
        quats[:, 0] = 1.0
        self.quats = nn.Parameter(quats)

        # Log-space scale for positive radii after exp().
        self.log_scales = nn.Parameter(torch.full((num_gaussians, 3), -2.3, device="cuda"))

        # Raw logits for alpha and RGB (activated by sigmoid).
        self.opacity_logits = nn.Parameter(torch.full((num_gaussians,), -1.0, device="cuda"))
        self.color_logits = nn.Parameter(torch.randn(num_gaussians, 3, device="cuda"))

    def activated_parameters(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        means = self.means
        quats = F.normalize(self.quats, dim=-1)
        scales = torch.exp(self.log_scales)
        opacities = torch.sigmoid(self.opacity_logits)
        colors = torch.sigmoid(self.color_logits)
        return means, quats, scales, opacities, colors

    def render(
        self,
        viewmats: torch.Tensor,
        Ks: torch.Tensor,
        width: int,
        height: int,
    ) -> torch.Tensor:

        means, quats, scales, opacities, colors = self.activated_parameters()
        rendered, _, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=width,
            height=height,
            sh_degree=None,
        )
        return rendered  # [B, H, W, 3]


class CameraParameterSampler:
    """Samples random camera parameters from configured ranges."""

    def __init__(self, ranges: CameraRanges) -> None:
        self.ranges = ranges

    def __call__(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        azimuth_deg = torch.empty(batch_size, device="cuda").uniform_(*self.ranges.azimuth_range)
        elevation_deg = torch.empty(batch_size, device="cuda").uniform_(*self.ranges.elevation_range)
        distance = torch.empty(batch_size, device="cuda").uniform_(*self.ranges.distance_range)
        return azimuth_deg, elevation_deg, distance
