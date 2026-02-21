import torch
import torch.nn.functional as F
from torch import nn
import timm
from timm.data import create_transform, resolve_model_data_config


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
            layer_loss = F.smooth_l1_loss(b_mean, r_mean) + F.smooth_l1_loss(b_var, r_var)
            weight = self.first_bn_multiplier if idx == 0 else 1.0
            loss = loss + weight * layer_loss

        total_loss = loss / len(self._bn_stats)
        self._bn_stats = []
        return total_loss


class FrozenBackbone(nn.Module):
    """Frozen pretrained timm backbone with model-specific eval preprocessing."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.data_config = resolve_model_data_config(backbone)
        self.preprocess = create_transform(**self.data_config, is_training=False)

        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.preprocess(images))


def create_backbone(name: str) -> FrozenBackbone:
    backbone = timm.create_model(name, pretrained=True)
    return FrozenBackbone(backbone=backbone)
