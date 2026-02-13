import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class FrozenStandardBackbone(nn.Module):
    """
    Wraps a backbone with ImageNet input standardization.
    Sets the backbone to evaluation mode
    Disables gradient computation.
    Moves to CUDA
    """

    def __init__(
        self,
        backbone: nn.Module,
    ) -> None:
        super().__init__()
        self.backbone = backbone.to(device="cuda")
        self.mean_tensor = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32, device="cuda").view(1, 3, 1, 1)
        self.std_tensor = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32, device="cuda").view(1, 3, 1, 1)

        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        normalized = (images - self.mean_tensor) / self.std_tensor
        return self.backbone(normalized)


def create_resnet18():
    return FrozenStandardBackbone(resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval())
