from gsplatviz.backbones import create_resnet18
import torch
from pathlib import Path
from tqdm.auto import tqdm

from gsplatviz.camera import CameraRanges, build_K, viewmats_from_spherical
from gsplatviz.splat import GaussianSplat, CameraParameterSampler
from gsplatviz.model_visualizer import GaussianSplatModelVisualizer


num_gaussians = 10000
width = 256
height = 256
steps = 3000000
lr = 1e-3
seed = 42
viz_every = 200
viz_every_factor = 1.1
vis_frames = 72
vis_fps = 20
target_class = 388
backbone = "resnet18"
batch_size = 16

camera_ranges = CameraRanges(
    azimuth_range=(-180.0, 180.0),
    elevation_range=(-10.0, 10.0),
    distance_range=(0.8, 1.2),
)
model_output_dir = Path("trained_gaussiansplat_models")


if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required")

torch.manual_seed(seed)
model_output_dir.mkdir(parents=True, exist_ok=True)

classifier = create_resnet18()

print(f"Training on cuda with {num_gaussians} gaussians")
visualization_dir = Path(f"training_{backbone}_{target_class}_visualizations")
visualization_dir.mkdir(parents=True, exist_ok=True)
model = GaussianSplat(num_gaussians=num_gaussians)
K = build_K(width, height)
sampler = CameraParameterSampler(ranges=camera_ranges)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

current_viz_every = max(1, int(viz_every))
next_viz_step = 1

progress = tqdm(range(1, steps + 1), desc=f"class={target_class}", total=steps)
for step in progress:
    optimizer.zero_grad(set_to_none=True)

    azimuth_deg, elevation_deg, distance = sampler(batch_size=batch_size)
    viewmats = viewmats_from_spherical(azimuth_deg, elevation_deg, distance)
    Ks = K.unsqueeze(0).expand(batch_size, -1, -1)
    rendered = model.render(viewmats=viewmats, Ks=Ks, width=width, height=height)

    rendered_rgb = rendered[..., :3].permute(0, 3, 1, 2).contiguous().clamp(0.0, 1.0)
    scores = classifier(rendered_rgb)
    loss = -scores[:, target_class].mean()

    loss.backward()
    optimizer.step()

    if step == next_viz_step or step == steps:
        iter_gif_path = visualization_dir / f"iter_{step:07d}.gif"
        visualizer = GaussianSplatModelVisualizer(
            model=model,
            width=width,
            height=height,
            ranges=camera_ranges,
        )
        visualizer.create_gif(
            output_path=str(iter_gif_path),
            num_frames=vis_frames,
            fps=vis_fps,
        )
        if step == next_viz_step:
            next_viz_step += current_viz_every
            current_viz_every = int(round(current_viz_every * viz_every_factor))

    if step % 10 == 1:
        progress.set_postfix(loss=f"{loss.item():.6f}")

final_model_path = model_output_dir / f"training_{backbone}_{target_class}.pt"
torch.save(
    {
        "backbone": backbone,
        "target_class": int(target_class),
        "state_dict": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
    },
    final_model_path,
)
print(f"Saved final GaussianSplatModel to {final_model_path}")

print("Done.")
