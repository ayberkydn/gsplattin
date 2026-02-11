from pathlib import Path

import numpy as np
import torch
from PIL import Image


def frame_from_render(rendered: torch.Tensor) -> np.ndarray:
    """Convert rendered tensor [1, H, W, 3] to uint8 image [H, W, 3]."""
    frame = rendered[0, ..., :3].detach().clamp(0.0, 1.0).mul(255.0).byte().cpu().numpy()
    return frame


def save_gif(frames: list[np.ndarray], output_path: str, fps: int) -> None:
    if not frames:
        return

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    images = [Image.fromarray(frame, mode="RGB") for frame in frames]
    duration_ms = max(1, int(1000 / max(1, fps)))
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )
