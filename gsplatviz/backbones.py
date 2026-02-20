import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import (
    DenseNet121_Weights,
    DenseNet201_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
    ViT_B_16_Weights,
    Wide_ResNet50_2_Weights,
    Wide_ResNet101_2_Weights,
    densenet121,
    densenet201,
    resnet18,
    resnet50,
    vit_b_16,
    wide_resnet50_2,
    wide_resnet101_2,
)


class ImageNormalizer(nn.Module):
    """Standard ImageNet normalization."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
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
    Wraps a backbone with ImageNet input standardization.
    Sets the backbone to evaluation mode and disables gradient computation.
    """

    def __init__(
        self,
        backbone: nn.Module,
        input_size: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.input_size = input_size
        self.normalizer = ImageNormalizer()

        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.input_size is not None:
            images = F.interpolate(
                images, size=self.input_size, mode="bilinear", align_corners=False
            )
        normalized = self.normalizer(images)
        return self.backbone(normalized)


def create_resnet18():
    return FrozenStandardBackbone(resnet18(weights=ResNet18_Weights.IMAGENET1K_V1))


def create_resnet50():
    return FrozenStandardBackbone(resnet50(weights=ResNet50_Weights.IMAGENET1K_V1))


def create_wideresnet50():
    return FrozenStandardBackbone(
        wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
    )


def create_densenet121():
    return FrozenStandardBackbone(densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1))


def create_wideresnet101():
    return FrozenStandardBackbone(
        wide_resnet101_2(weights=Wide_ResNet101_2_Weights.IMAGENET1K_V1)
    )




def create_densenet201():
    return FrozenStandardBackbone(densenet201(weights=DenseNet201_Weights.IMAGENET1K_V1))


def create_vit_b_16():
    return FrozenStandardBackbone(
        vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1),
        input_size=(224, 224),
    )


BACKBONES = {
    "resnet18": create_resnet18,
    "resnet50": create_resnet50,
    "wideresnet50": create_wideresnet50,
    "wideresnet101": create_wideresnet101,
    "densenet121": create_densenet121,
    "densenet201": create_densenet201,
    "vit_b_16": create_vit_b_16,
}


def create_backbone(name: str) -> FrozenStandardBackbone:
    if name not in BACKBONES:
        raise ValueError(
            f"Unknown backbone '{name}'. Available: {list(BACKBONES.keys())}"
        )
    return BACKBONES[name]()
