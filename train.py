from dataclasses import asdict, dataclass
import json
from pathlib import Path

import tyro

import torch
import torch.nn.functional as F
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required")

torch.set_default_device("cuda")
torch.set_float32_matmul_precision("high")
from tqdm.auto import tqdm
import wandb

from gsplat.strategy import MCMCStrategy
from gsplatviz.backbones import BNMatchingLoss, create_backbone
from gsplatviz.camera import CameraRanges, build_K, viewmats_from_spherical
from gsplatviz.splat_visualizer import SplatVizualizer
from gsplatviz.splat import CameraParameterSampler, GaussianSplat


@dataclass(frozen=True)
class GaussianConfig:
    init_count: int = 5000
    max_count: int = 10000
    sh_degree: int = 1


@dataclass(frozen=True)
class CameraConfig:
    width: int = 256
    height: int = 256
    azimuth_range: float = 90.0
    elevation_range: float = 60.0
    distance_range: tuple[float, float] = (0.7, 1.3)


@dataclass(frozen=True)
class OptimizationConfig:
    seed: int = 42

    steps: int = 20000
    batch_size: int = 32

    means_lr: float = 1.6e-4
    means_lr_final: float = 1.6e-6
    sh_lr: float = 2.5e-3
    opacities_lr: float = 0.05
    scales_lr: float = 0.005
    quats_lr: float = 0.001


@dataclass(frozen=True)
class LossConfig:
    score_weight: float = 1.0
    prob_weight: float = 0.0
    bn_weight: float = 0.1
    first_bn_multiplier: float = 5.0
    scale_reg: float = 0.01
    opacity_reg: float = 0.01


@dataclass(frozen=True)
class LoggingConfig:
    viz_every: int = 500
    viz_every_factor: float = 1.2
    vis_frames: int = 100
    vis_fps: int = 20
    output_dir: str = "trained_gaussiansplat_models"
    wandb_project: str = "gsplattin"


@dataclass(frozen=True)
class TrainConfig:
    target_class: int = 950
    backbone: str = "resnet18"

    gs: GaussianConfig = GaussianConfig()
    camera: CameraConfig = CameraConfig()
    optim: OptimizationConfig = OptimizationConfig()
    loss: LossConfig = LossConfig()
    logging: LoggingConfig = LoggingConfig()

    def to_camera_ranges(self) -> CameraRanges:
        return CameraRanges(
            azimuth_range=self.camera.azimuth_range,
            elevation_range=self.camera.elevation_range,
            distance_range=self.camera.distance_range,
        )


def save_config(config: TrainConfig, output_dir: Path, backbone: str, target_class: int) -> Path:
    config_path = output_dir / f"training_{backbone}_{target_class}.config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2, sort_keys=True)
    return config_path


def main() -> None:
    config = tyro.cli(TrainConfig)
    torch.manual_seed(config.optim.seed)

    wandb.init(
        project=config.logging.wandb_project,
        config=asdict(config),
    )


    model_output_dir = Path(config.logging.output_dir)
    model_output_dir.mkdir(parents=True, exist_ok=True)

    camera_ranges = config.to_camera_ranges()

    classifier = torch.compile(create_backbone(config.backbone))
    bn_matching_loss = BNMatchingLoss(
        create_backbone("resnet18"),
        first_bn_multiplier=config.loss.first_bn_multiplier,
    )
    bn_matching_loss.model = torch.compile(bn_matching_loss.model)

    print(f"Training on cuda with {config.gs.init_count} initial gaussians (max {config.gs.max_count})")
    visualization_dir = Path("visualizations") / config.backbone / str(config.target_class)
    visualization_dir.mkdir(parents=True, exist_ok=True)

    model = GaussianSplat(num_gaussians=config.gs.init_count, sh_degree=config.gs.sh_degree)
    K = build_K(config.camera.width, config.camera.height)
    sampler = CameraParameterSampler(ranges=camera_ranges)

    strategy = MCMCStrategy(
        cap_max=config.gs.max_count,
        noise_lr=0.01,
        refine_start_iter=0,
        refine_every=200,
    )
    strategy_state = strategy.initialize_state()
    lrs = {
        "means": config.optim.means_lr,
        "sh_coeffs": config.optim.sh_lr,
        "opacities": config.optim.opacities_lr,
        "scales": config.optim.scales_lr,
        "quats": config.optim.quats_lr,
    }
    optimizers = {
        name: torch.optim.Adam([param], lr=lrs[name])
        for name, param in model.params.items()
    }
    strategy.check_sanity(model.params, optimizers)
    target_labels = torch.full((config.optim.batch_size,), config.target_class, dtype=torch.long)

    saved_config_path = save_config(config, model_output_dir, config.backbone, config.target_class)
    print(f"Saved config to {saved_config_path}")

    current_viz_every = max(1, int(config.logging.viz_every))
    next_viz_step = 1

    progress = tqdm(range(1, config.optim.steps + 1), desc=f"class={config.target_class}", total=config.optim.steps, mininterval=5.0)
    for step in progress:
        # Exponential learning rate decay for means
        lr_ratio = (step - 1) / (config.optim.steps - 1)
        curr_means_lr = config.optim.means_lr * (config.optim.means_lr_final / config.optim.means_lr) ** lr_ratio
        for param_group in optimizers["means"].param_groups:
            param_group["lr"] = curr_means_lr

        for opt in optimizers.values():
            opt.zero_grad(set_to_none=True)

        azimuth_deg, elevation_deg, distance = sampler(batch_size=config.optim.batch_size)
        viewmats = viewmats_from_spherical(azimuth_deg, elevation_deg, distance)
        Ks = K.unsqueeze(0).expand(config.optim.batch_size, -1, -1)

        rendered = model.render(
            viewmats=viewmats,
            Ks=Ks,
            width=config.camera.width,
            height=config.camera.height,
        )

        rendered_rgb = rendered[..., :3].permute(0, 3, 1, 2).contiguous().clamp(0.0, 1.0)
        ##TODO consider sigmoid

        bn_loss = bn_matching_loss(rendered_rgb)

        with torch.amp.autocast("cuda"):
            scores = classifier(rendered_rgb)
        scores = scores.float()
        score_loss = -scores[:, config.target_class].mean()
        prob_loss = F.cross_entropy(scores, target_labels)

        loss = config.loss.score_weight * score_loss + config.loss.prob_weight * prob_loss + config.loss.bn_weight * bn_loss

        # Regularization from 3DGS-MCMC
        _, _, scales, opacities, _ = model.activated_parameters()
        reg_loss = config.loss.scale_reg * scales.mean() + config.loss.opacity_reg * opacities.mean()
        loss = loss + reg_loss

        strategy.step_pre_backward(model.params, optimizers, strategy_state, step, {})
        loss.backward()
        strategy.step_post_backward(model.params, optimizers, strategy_state, step, {}, lr=curr_means_lr)

        for opt in optimizers.values():
            opt.step()


        if step == next_viz_step or step == config.optim.steps:
            iter_gif_path = visualization_dir / f"iter_{step:07d}.gif"
            visualizer = SplatVizualizer(
                splat=model,
                width=config.camera.width,
                height=config.camera.height,
                ranges=camera_ranges,
            )
            visualizer.create_gif(
                output_path=str(iter_gif_path),
                num_frames=config.logging.vis_frames,
                fps=config.logging.vis_fps,
            )
            wandb.log({
                "viz/evolution": wandb.Video(str(iter_gif_path), format="webm")
            }, step=step)
            tqdm.write(f"Saved visualization to {iter_gif_path}")
            if step == next_viz_step:
                next_viz_step += current_viz_every
                current_viz_every = int(round(current_viz_every * config.logging.viz_every_factor))

        if step % 50 == 1:
            wandb.log({
                "loss/total": loss.item(),
                "loss/score": score_loss.item(),
                "loss/prob": prob_loss.item(),
                "loss/bn": bn_loss.item(),
                "loss/reg": reg_loss.item(),
                "params/n_gaussians": model.params["means"].shape[0],
                "params/means_lr": curr_means_lr,
            }, step=step)

            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                score=f"{score_loss.item():.4f}",
                prob=f"{prob_loss.item():.4f}",
                bn=f"{bn_loss.item():.4f}",
                reg=f"{reg_loss.item():.4f}",
                n_gs=str(model.params["means"].shape[0]),
            )


    wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()
