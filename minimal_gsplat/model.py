import torch
import torch.nn.functional as F
from gsplat import rasterization
from torch import nn



class GaussianSplatModel(nn.Module):
    """Minimal Gaussian splat model with only core learnable parameters."""

    def __init__(self, num_gaussians: int, device: torch.device) -> None:
        super().__init__()

        # Start near the camera frustum so the initial render is not empty.
        means = torch.randn(num_gaussians, 3, device=device) * 0.25
        means[:, 2] += 2.5
        self.means = nn.Parameter(means)

        # Quaternion rotation (w, x, y, z). Initialized to identity.
        quats = torch.zeros(num_gaussians, 4, device=device)
        quats[:, 0] = 1.0
        self.quats = nn.Parameter(quats)

        # Log-space scale for positive radii after exp().
        self.log_scales = nn.Parameter(torch.full((num_gaussians, 3), -2.3, device=device))

        # Raw logits for alpha and RGB (activated by sigmoid).
        self.opacity_logits = nn.Parameter(torch.full((num_gaussians,), -1.0, device=device))
        self.color_logits = nn.Parameter(torch.randn(num_gaussians, 3, device=device))

    def activated_parameters(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        means = self.means
        quats = F.normalize(self.quats, dim=-1)
        scales = torch.exp(self.log_scales)
        opacities = torch.sigmoid(self.opacity_logits)
        colors = torch.sigmoid(self.color_logits)
        return means, quats, scales, opacities, colors

    def render_rgb(self, viewmats: torch.Tensor, Ks: torch.Tensor, width: int, height: int) -> torch.Tensor:
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
        return rendered  # [C, H, W, 3] for one camera: [1, H, W, 3]
