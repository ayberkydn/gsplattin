import torch
import math


class Camera:
    """Single pinhole camera with FOV-based intrinsics."""

    def __init__(
        self,
        width: int,
        height: int,
        horizontal_fov_deg: float,
        vertical_fov_deg: float,
        device: torch.device,
        world_to_camera: torch.Tensor | None = None,
    ) -> None:

        self.width = width
        self.height = height
        self.horizontal_fov_deg = horizontal_fov_deg
        self.vertical_fov_deg = vertical_fov_deg
        self.device = device

        if world_to_camera is None:
            world_to_camera = torch.eye(4, dtype=torch.float32, device=device)
        else:
            world_to_camera = world_to_camera.to(dtype=torch.float32, device=device)
            if world_to_camera.shape != (4, 4):
                raise ValueError("world_to_camera must have shape [4, 4].")

        self._viewmats = world_to_camera.unsqueeze(0)  # [1, 4, 4]
        self._Ks = self._build_intrinsics().unsqueeze(0)  # [1, 3, 3]

    def _build_intrinsics(self) -> torch.Tensor:
        fx = self.width / (2.0 * math.tan(math.radians(self.horizontal_fov_deg) * 0.5))
        fy = self.height / (2.0 * math.tan(math.radians(self.vertical_fov_deg) * 0.5))
        cx = self.width * 0.5
        cy = self.height * 0.5
        K = torch.eye(3, dtype=torch.float32, device=self.device)
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = cx
        K[1, 2] = cy
        return K



    def matrices(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._viewmats, self._Ks
