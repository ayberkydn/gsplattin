import math

import torch

from minimal_gsplat.camera import Camera
from minimal_gsplat.gif_utils import frame_from_render, save_gif
from minimal_gsplat.model import GaussianSplatModel


num_gaussians = 512
width = 320
height = 240
horizontal_fov_deg = 60.0
vertical_fov_deg = 45.0
steps = 1000
lr = 5e-3
seed = 0
print_every = 100
gif_path = "training.gif"
gif_every = 10
gif_fps = 5

# Rotating camera GIF settings
create_rotation_gif = True
rotation_gif_path = "final_splat_rotation.gif"
rotation_gif_fps = 10
rotation_radius = 4.0
rotation_frames = 60  # Number of frames for full 360 rotation

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required")

device = torch.device("cuda")
torch.manual_seed(seed)

model = GaussianSplatModel(num_gaussians=num_gaussians, device=device)
camera = Camera(
    width=width,
    height=height,
    horizontal_fov_deg=horizontal_fov_deg,
    vertical_fov_deg=vertical_fov_deg,
    device=device,
)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
frames: list = []

print(f"Training on {device} with {num_gaussians} gaussians")
for step in range(1, steps + 1):
    optimizer.zero_grad(set_to_none=True)

    viewmats, Ks = camera.matrices()
    rendered = model.render_rgb(viewmats, Ks, width, height)

    loss = rendered.mean()

    if gif_path and (step == 1 or step % gif_every == 0 or step == steps):
        frames.append(frame_from_render(rendered))

    loss.backward()
    optimizer.step()

    if step == 1 or step % print_every == 0:
        print(f"step={step:04d} loss={loss.item():.6f}")

if gif_path:
    save_gif(frames, gif_path, fps=gif_fps)
    print(f"Saved training GIF to {gif_path}")

# Create rotating camera GIF of final splat
if create_rotation_gif:
    print(f"Creating rotating camera GIF with {rotation_frames} frames...")
    rotation_frames_list = []

    for i in range(rotation_frames):
        angle = 2 * math.pi * i / rotation_frames
        # Camera position on circle in XZ plane
        eye_x = rotation_radius * math.cos(angle)
        eye_z = rotation_radius * math.sin(angle)
        eye_y = 0.0  # Camera height
        eye = torch.tensor([eye_x, eye_y, eye_z], dtype=torch.float32, device=device)

        # Target to look at (origin)
        target = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)

        # Up vector
        up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)

        # Compute camera coordinate system
        forward = target - eye
        forward = forward / forward.norm()

        right = torch.cross(forward, up)
        right = right / right.norm()

        new_up = torch.cross(right, forward)

        # Build rotation matrix (camera to world)
        R = torch.stack([right, new_up, -forward], dim=1)  # [3, 3]

        # Build translation
        t = -R @ eye  # [3]

        # Build 4x4 world_to_camera matrix
        world_to_camera = torch.eye(4, dtype=torch.float32, device=device)
        world_to_camera[:3, :3] = R
        world_to_camera[:3, 3] = t

        # Create camera and render
        rot_camera = Camera(
            width=width,
            height=height,
            horizontal_fov_deg=horizontal_fov_deg,
            vertical_fov_deg=vertical_fov_deg,
            device=device,
            world_to_camera=world_to_camera,
        )

        viewmats, Ks = rot_camera.matrices()
        rendered = model.render_rgb(viewmats, Ks, width, height)
        rotation_frames_list.append(frame_from_render(rendered))

    save_gif(rotation_frames_list, rotation_gif_path, fps=rotation_gif_fps)
    print(f"Saved rotating camera GIF to {rotation_gif_path}")

print("Done.")
