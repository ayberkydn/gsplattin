import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import (
    DenseNet121_Weights,
    DenseNet161_Weights,
    DenseNet169_Weights,
    DenseNet201_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
    ViT_B_16_Weights,
    Wide_ResNet50_2_Weights,
    Wide_ResNet101_2_Weights,
    densenet121,
    densenet161,
    densenet169,
    densenet201,
    resnet18,
    resnet50,
    vit_b_16,
    wide_resnet50_2,
    wide_resnet101_2,
)


class FrozenStandardBackbone(nn.Module):
    """
    Wraps a backbone with ImageNet input standardization.
    Sets the backbone to evaluation mode
    Disables gradient computation.
    registers batcnorm stats
    """

    def __init__(
        self,
        backbone: nn.Module,
        input_size: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.input_size = input_size
        self.mean_tensor = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1)
        self.std_tensor = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1)

        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        self._bn_stats: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for module in self.backbone.modules():
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
        self._bn_stats.append((batch_mean, batch_var, module.running_mean, module.running_var))

    def bn_matching_loss(self, first_bn_multiplier: float) -> torch.Tensor:
        """MSE between batch statistics and BN running statistics. Call after forward().
        DeepInversion-style weighting: upweight first BN layer by `first_bn_multiplier`.
        """
        if not self._bn_stats:
            return torch.tensor(0.0, device=self.mean_tensor.device)

        loss = torch.tensor(0.0, device=self.mean_tensor.device)
        for idx, (batch_mean, batch_var, running_mean, running_var) in enumerate(self._bn_stats):
            layer_loss = F.mse_loss(batch_mean, running_mean) + F.mse_loss(batch_var, running_var)
            weight = first_bn_multiplier if idx < 3 else 1.0
            loss = loss + weight * layer_loss
        return loss / len(self._bn_stats)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self._bn_stats = []
        if self.input_size is not None:
            target_height, target_width = self.input_size
            images = F.interpolate(images, size=(target_height, target_width), mode="bilinear", align_corners=False)
        else:
            "woweeeee, this is a bit of a hack to avoid having to register hooks for the resizing layers in vit_b_16"

        normalized = (images - self.mean_tensor) / self.std_tensor
        return self.backbone(normalized)


def create_resnet18():
    return FrozenStandardBackbone(resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval())


def create_resnet50():
    return FrozenStandardBackbone(resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).eval())


def create_wideresnet50():
    return FrozenStandardBackbone(wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1).eval())


def create_densenet121():
    return FrozenStandardBackbone(densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1).eval())


def create_wideresnet101():
    return FrozenStandardBackbone(wide_resnet101_2(weights=Wide_ResNet101_2_Weights.IMAGENET1K_V1).eval())


def create_densenet161():
    return FrozenStandardBackbone(densenet161(weights=DenseNet161_Weights.IMAGENET1K_V1).eval())


def create_densenet169():
    return FrozenStandardBackbone(densenet169(weights=DenseNet169_Weights.IMAGENET1K_V1).eval())


def create_densenet201():
    return FrozenStandardBackbone(densenet201(weights=DenseNet201_Weights.IMAGENET1K_V1).eval())


def create_vit_b_16():
    return FrozenStandardBackbone(
        vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).eval(),
        input_size=(224, 224),
    )


BACKBONES = {
    "resnet18": create_resnet18,
    "resnet50": create_resnet50,
    "wideresnet50": create_wideresnet50,
    "wideresnet101": create_wideresnet101,
    "densenet121": create_densenet121,
    "densenet161": create_densenet161,
    "densenet169": create_densenet169,
    "densenet201": create_densenet201,
    "vit_b_16": create_vit_b_16,
}


def create_backbone(name: str) -> FrozenStandardBackbone:
    if name not in BACKBONES:
        raise ValueError(f"Unknown backbone '{name}'. Available: {list(BACKBONES.keys())}")
    return BACKBONES[name]()
