from dataclasses import asdict, dataclass
import json
from pathlib import Path

import tyro

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from gsplat.strategy import MCMCStrategy
from gsplatviz.backbones import create_backbone
from gsplatviz.camera import CameraRanges, build_K, viewmats_from_spherical
from gsplatviz.model_visualizer import SplatVizualizer
from gsplatviz.splat import CameraParameterSampler, GaussianSplat


@dataclass
class TrainConfig:
    num_gaussians: int = 10000
    width: int = 256
    height: int = 256
    steps: int = 30000
    means_lr: float = 1.6e-4
    means_lr_final: float = 1.6e-6
    sh_lr: float = 2.5e-3
    opacities_lr: float = 0.05
    scales_lr: float = 0.005
    quats_lr: float = 0.001
    seed: int = 42
    viz_every: int = 500
    viz_every_factor: float = 1.1
    vis_frames: int = 72
    vis_fps: int = 20
    target_class: int = 388
    backbone: str = "resnet18"
    batch_size: int = 4

    score_loss_weight: float = 0.1
    bn_loss_weight: float = 0.0001
    prob_loss_weight: float = 1.0
    azimuth_min: float = -180.0
    azimuth_max: float = 180.0
    elevation_min: float = -10.0
    elevation_max: float = 10.0
    distance_min: float = 0.8
    distance_max: float = 1.2
    model_output_dir: str = "trained_gaussiansplat_models"

    def to_camera_ranges(self) -> CameraRanges:
        return CameraRanges(
            azimuth_range=(self.azimuth_min, self.azimuth_max),
            elevation_range=(self.elevation_min, self.elevation_max),
            distance_range=(self.distance_min, self.distance_max),
        )


def parse_args() -> TrainConfig:
    return tyro.cli(TrainConfig)


def save_config(config: TrainConfig, output_dir: Path, backbone: str, target_class: int) -> Path:
    config_path = output_dir / f"training_{backbone}_{target_class}.config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2, sort_keys=True)
    return config_path


def main() -> None:
    config = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.set_default_device("cuda")
    torch.manual_seed(config.seed)

    model_output_dir = Path(config.model_output_dir)
    model_output_dir.mkdir(parents=True, exist_ok=True)

    camera_ranges = config.to_camera_ranges()

    classifier = create_backbone(config.backbone)

    print(f"Training on cuda with {config.num_gaussians} gaussians")
    visualization_dir = Path(f"training_{config.backbone}_{config.target_class}_visualizations")
    visualization_dir.mkdir(parents=True, exist_ok=True)

    model = GaussianSplat(num_gaussians=config.num_gaussians)
    K = build_K(config.width, config.height)
    sampler = CameraParameterSampler(ranges=camera_ranges)

    strategy = MCMCStrategy(

    )
    strategy_state = strategy.initialize_state()
    lrs = {
        "means": config.means_lr,
        "sh_coeffs": config.sh_lr,
        "opacities": config.opacities_lr,
        "scales": config.scales_lr,
        "quats": config.quats_lr,
    }
    optimizers = {
        name: torch.optim.Adam([param], lr=lrs[name])
        for name, param in model.params.items()
    }
    strategy.check_sanity(model.params, optimizers)
    target_labels = torch.full((config.batch_size,), config.target_class, dtype=torch.long)

    saved_config_path = save_config(config, model_output_dir, config.backbone, config.target_class)
    print(f"Saved config to {saved_config_path}")

    current_viz_every = max(1, int(config.viz_every))
    next_viz_step = 1

    progress = tqdm(range(1, config.steps + 1), desc=f"class={config.target_class}", total=config.steps)
    for step in progress:
        # Exponential learning rate decay for means
        lr_ratio = (step - 1) / (config.steps - 1)
        curr_means_lr = config.means_lr * (config.means_lr_final / config.means_lr) ** lr_ratio
        for param_group in optimizers["means"].param_groups:
            param_group["lr"] = curr_means_lr

        for opt in optimizers.values():
            opt.zero_grad(set_to_none=True)

        azimuth_deg, elevation_deg, distance = sampler(batch_size=config.batch_size)
        viewmats = viewmats_from_spherical(azimuth_deg, elevation_deg, distance)
        Ks = K.unsqueeze(0).expand(config.batch_size, -1, -1)

        rendered, info = model.render(
            viewmats=viewmats,
            Ks=Ks,
            width=config.width,
            height=config.height,
        )

        rendered_rgb = rendered[..., :3].permute(0, 3, 1, 2).contiguous().clamp(0.0, 1.0)
        scores = classifier(rendered_rgb)

        score_loss = -scores[:, config.target_class].mean()
        prob_loss = F.cross_entropy(scores, target_labels)
        bn_loss = classifier.bn_matching_loss()

        loss = config.score_loss_weight * score_loss + config.prob_loss_weight * prob_loss + config.bn_loss_weight * bn_loss

        strategy.step_pre_backward(model.params, optimizers, strategy_state, step, info)
        loss.backward()
        strategy.step_post_backward(model.params, optimizers, strategy_state, step, info, lr=curr_means_lr)

        for opt in optimizers.values():
            opt.step()

        if step == next_viz_step or step == config.steps:
            iter_gif_path = visualization_dir / f"iter_{step:07d}.gif"
            visualizer = SplatVizualizer(
                splat=model,
                width=config.width,
                height=config.height,
                ranges=camera_ranges,
            )
            visualizer.create_gif(
                output_path=str(iter_gif_path),
                num_frames=config.vis_frames,
                fps=config.vis_fps,
            )
            tqdm.write(f"Saved visualization to {iter_gif_path}")
            if step == next_viz_step:
                next_viz_step += current_viz_every
                current_viz_every = int(round(current_viz_every * config.viz_every_factor))

        if step % 10 == 1:
            progress.set_postfix(
                loss=f"{loss.item():.6f}",
                score_loss=f"{score_loss.item():.6f}",
                bn_loss=f"{bn_loss.item():.6f}",
            )

    final_model_path = model_output_dir / f"training_{config.backbone}_{config.target_class}.pt"
    torch.save(
        {
            "backbone": config.backbone,
            "target_class": int(config.target_class),
            "config": asdict(config),
            "state_dict": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        },
        final_model_path,
    )
    print(f"Saved final GaussianSplatModel to {final_model_path}")

    final_ply_path = model_output_dir / f"training_{config.backbone}_{config.target_class}.ply"
    model.export_ply(str(final_ply_path))
    print(f"Saved final PLY to {final_ply_path}")

    print("Done.")


if __name__ == "__main__":
    main()
