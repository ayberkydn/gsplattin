import torch
import torch.nn.functional as F
from gsplat import export_splats, rasterization
from torch import nn

from .camera import CameraRanges


class GaussianSplat(nn.Module):
    """Gaussian splat with learnable parameters and batched rendering."""

    def __init__(self, num_gaussians: int, sh_degree: int) -> None:
        super().__init__()

        self.sh_degree = int(sh_degree)
        self.num_sh_coeffs = (self.sh_degree + 1) ** 2

        # Initialize gaussians around the origin.
        means = torch.randn(num_gaussians, 3, device="cuda") * 0.1

        # Quaternion rotation (w, x, y, z). Initialized to identity.
        quats = torch.zeros(num_gaussians, 4, device="cuda")
        quats[:, 0] = 1.0

        # Log-space scale for positive radii after exp().
        scales = torch.full((num_gaussians, 3), -2.3, device="cuda")

        # Raw logits for alpha and SH coefficients.
        opacities = torch.full((num_gaussians,), -1.0, device="cuda")
        sh_coeffs = torch.zeros(num_gaussians, self.num_sh_coeffs, 3, device="cuda")
        with torch.no_grad():
            # Initialize SH0 from a random RGB prior to keep initial renders visible.
            sh_coeffs[:, 0, :] = torch.randn(num_gaussians, 3, device="cuda")

        self.params = nn.ParameterDict(
            {
                "means": nn.Parameter(means),
                "quats": nn.Parameter(quats),
                "scales": nn.Parameter(scales),
                "opacities": nn.Parameter(opacities),
                "sh_coeffs": nn.Parameter(sh_coeffs),
            }
        )

    def activated_parameters(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        means = self.params["means"]
        quats = F.normalize(self.params["quats"], dim=-1)
        scales = torch.exp(self.params["scales"]).clamp(max=10.0)
        opacities = torch.sigmoid(self.params["opacities"])
        # Keep SH coefficients unconstrained; gsplat evaluates SH basis internally.
        sh_coeffs = self.params["sh_coeffs"]
        return means, quats, scales, opacities, sh_coeffs

    def render(
        self,
        viewmats: torch.Tensor,
        Ks: torch.Tensor,
        width: int,
        height: int,
    ) -> torch.Tensor:
        """Render the gaussians to images."""
        means, quats, scales, opacities, colors = self.activated_parameters()

        # Using packed=False for faster batch rendering as per gsplat documentation.
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
            sh_degree=self.sh_degree,
            packed=False,
        )
        return rendered

    def export_ply(self, path: str) -> None:
        """Export the gaussians to a PLY file."""
        means, quats, scales, opacities, sh_coeffs = self.activated_parameters()
        # SH export expects SH0 and higher-order coefficients separately.
        sh0 = sh_coeffs[:, :1, :]
        shN = sh_coeffs[:, 1:, :]
        export_splats(
            means=means,
            scales=scales,
            quats=quats,
            opacities=opacities,
            sh0=sh0,
            shN=shN,
            format="ply",
            save_to=path,
        )


class CameraParameterSampler:
    """Samples random camera parameters from configured ranges."""

    def __init__(self, ranges: CameraRanges) -> None:
        self.ranges = ranges

    def __call__(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        azimuth_deg = torch.empty(batch_size, device="cuda").uniform_(
            -self.ranges.azimuth_range, self.ranges.azimuth_range
        )
        elevation_deg = torch.empty(batch_size, device="cuda").uniform_(
            -self.ranges.elevation_range, self.ranges.elevation_range
        )
        distance = torch.empty(batch_size, device="cuda").uniform_(*self.ranges.distance_range)
        return azimuth_deg, elevation_deg, distance
