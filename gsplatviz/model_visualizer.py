import math

import numpy as np
import torch

from .gif_utils import frame_from_render, save_gif
from .camera import CameraRanges, build_K, viewmats_from_spherical
from .splat import GaussianSplat


class SplatVizualizer:
    """Renders a GaussianSplat across camera ranges and saves a GIF."""

    def __init__(
        self,
        splat: GaussianSplat,
        width: int,
        height: int,
        ranges: CameraRanges,
    ) -> None:
        self.splat = splat
        self.width = width
        self.height = height
        self.ranges = ranges
        self.K = build_K(width, height)

    def _periodic_interp(
        self,
        min_value: float,
        max_value: float,
        phase: torch.Tensor,
        phase_shift: float = 0.0,
    ) -> torch.Tensor:
        if min_value == max_value:
            return torch.full_like(phase, min_value, dtype=torch.float32)
        alpha = 0.5 * (torch.sin(phase + phase_shift - (math.pi * 0.5)) + 1.0)
        return min_value + (max_value - min_value) * alpha

    def _camera_trajectory(self, num_frames: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        phase = torch.linspace(0.0, 2.0 * math.pi, steps=num_frames + 1, dtype=torch.float32)[:-1]
        r = self.ranges
        azimuth_deg = self._periodic_interp(r.azimuth_range[0], r.azimuth_range[1], phase, phase_shift=0.0)
        elevation_deg = self._periodic_interp(r.elevation_range[0], r.elevation_range[1], phase, phase_shift=math.pi * 0.5)
        distance = self._periodic_interp(r.distance_range[0], r.distance_range[1], phase, phase_shift=math.pi)
        return azimuth_deg, elevation_deg, distance

    def render_frames(self, num_frames: int) -> list[np.ndarray]:
        azimuth_deg, elevation_deg, distance = self._camera_trajectory(num_frames=num_frames)
        with torch.no_grad():
            viewmats = viewmats_from_spherical(azimuth_deg, elevation_deg, distance)
            Ks = self.K.unsqueeze(0).expand(num_frames, -1, -1)
            rendered_batch, _ = self.splat.render(viewmats=viewmats, Ks=Ks, width=self.width, height=self.height)
            frames = [frame_from_render(rendered_batch[idx : idx + 1]) for idx in range(num_frames)]
        return frames

    def create_gif(
        self,
        output_path: str,
        num_frames: int = 120,
        fps: int = 24,
    ) -> list[np.ndarray]:
        frames = self.render_frames(num_frames=num_frames)
        save_gif(frames, output_path=output_path, fps=fps)
        return frames
