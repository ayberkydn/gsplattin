import torch
import torch.nn.functional as F
from torch import nn
import timm
from timm.data import resolve_model_data_config


class ImageNormalizer(nn.Module):
    """Model-specific input normalization."""

    def __init__(
        self,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
    ) -> None:
        super().__init__()
        self.register_buffer(
            "mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class BNMatchingLoss(nn.Module):
    """
    MSE between batch statistics and running statistics.
    """

    def __init__(self, model: nn.Module, first_bn_multiplier: float = 1.0) -> None:
        super().__init__()
        self.model = model
        self.first_bn_multiplier = first_bn_multiplier
        self._bn_stats: list[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = []
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.register_forward_hook(self._bn_hook)

    def _bn_hook(
        self,
        module: nn.BatchNorm2d,
        input: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        x = input[0]
        batch_mean = x.mean(dim=(0, 2, 3))
        batch_var = x.var(dim=(0, 2, 3))
        assert module.running_mean is not None and module.running_var is not None
        self._bn_stats.append(
            (batch_mean, batch_var, module.running_mean, module.running_var)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Runs the model on images and returns the BN matching loss."""
        self._bn_stats = []
        self.model(images)

        if not self._bn_stats:
            return torch.tensor(0.0, device=images.device)

        device = self._bn_stats[0][0].device
        loss = torch.tensor(0.0, device=device)

        for idx, (b_mean, b_var, r_mean, r_var) in enumerate(self._bn_stats):
            layer_loss = F.mse_loss(b_mean, r_mean) + F.mse_loss(b_var, r_var)
            weight = self.first_bn_multiplier if idx < 3 else 1.0
            loss = loss + weight * layer_loss

        total_loss = loss / len(self._bn_stats)
        self._bn_stats = []
        return total_loss


class FrozenStandardBackbone(nn.Module):
    """
    Wraps a backbone with model-specific input preprocessing.
    Sets the backbone to evaluation mode and disables gradient computation.
    """

    def __init__(
        self,
        backbone: nn.Module,
        input_size: tuple[int, int],
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        interpolation: str,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.input_size = input_size
        self.interpolation = interpolation
        self.normalizer = ImageNormalizer(mean=mean, std=std)

        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if tuple(images.shape[-2:]) != self.input_size:
            images = F.interpolate(
                images,
                size=self.input_size,
                mode=self.interpolation,
                align_corners=False,
            )
        normalized = self.normalizer(images)
        return self.backbone(normalized)



def create_backbone(name: str) -> FrozenStandardBackbone:
    backbone = timm.create_model(name, pretrained=True)
    data_cfg = resolve_model_data_config(backbone)
    interpolation = data_cfg.get("interpolation", "bilinear")
    if interpolation not in {"bilinear", "bicubic"}:
        interpolation = "bilinear"

    return FrozenStandardBackbone(
        backbone=backbone,
        input_size=tuple(data_cfg["input_size"][-2:]),
        mean=tuple(data_cfg["mean"]),
        std=tuple(data_cfg["std"]),
        interpolation=interpolation,
    )
