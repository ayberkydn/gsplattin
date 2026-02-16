from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class CameraRanges:
    """Min/max bounds for spherical camera parameters."""

    azimuth_range: float = 180.0
    elevation_range: float = 10.0
    distance_range: tuple[float, float] = (0.8, 1.2)


def build_K(width: int, height: int) -> torch.Tensor:
    """Build a 3x3 intrinsic matrix assuming 90° FOV."""
    K = torch.eye(3, dtype=torch.float32, device="cuda")
    K[0, 0] = width * 0.5  # fx
    K[1, 1] = height * 0.5  # fy
    K[0, 2] = width * 0.5  # cx
    K[1, 2] = height * 0.5  # cy
    return K


def viewmats_from_spherical(
    azimuth_deg: torch.Tensor,
    elevation_deg: torch.Tensor,
    distance: torch.Tensor,
) -> torch.Tensor:
    """Compute batched 4x4 world-to-camera matrices from spherical coordinates.

    The camera always looks at the world origin.
    """
    batch_size = azimuth_deg.shape[0]

    az = torch.deg2rad(azimuth_deg.float())
    el = torch.deg2rad(elevation_deg.float())
    dist = distance.float()

    eye = torch.stack(
        [
            dist * torch.cos(el) * torch.sin(az),
            dist * torch.sin(el),
            dist * torch.cos(el) * torch.cos(az),
        ],
        dim=-1,
    )

    target = torch.zeros_like(eye)
    up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device="cuda").expand(batch_size, -1)

    forward = F.normalize(target - eye, dim=-1)
    right = F.normalize(torch.cross(forward, up, dim=-1), dim=-1)
    cam_up = torch.cross(right, forward, dim=-1)

    # Build camera-to-world: columns are camera axes, translation is eye.
    C2W = torch.eye(4, dtype=torch.float32, device="cuda").unsqueeze(0).repeat(batch_size, 1, 1)
    C2W[:, :3, 0] = right
    C2W[:, :3, 1] = cam_up
    C2W[:, :3, 2] = forward
    C2W[:, :3, 3] = eye

    # Invert rigid-body transform: R^T and -R^T @ eye.
    R_inv = C2W[:, :3, :3].transpose(1, 2)
    t_inv = torch.bmm(R_inv, (-eye).unsqueeze(-1)).squeeze(-1)

    W2C = torch.eye(4, dtype=torch.float32, device="cuda").unsqueeze(0).repeat(batch_size, 1, 1)
    W2C[:, :3, :3] = R_inv
    W2C[:, :3, 3] = t_inv
    return W2C
