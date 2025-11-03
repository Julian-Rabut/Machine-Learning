import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18FeatureExtractor(nn.Module):
    """
    Frozen ResNet18 backbone to extract a single 512-d global feature per image.
    - We take the output after global average pooling (before the final FC).
    - This is simple and works well for image-level anomaly detection via distances.
    """
    def __init__(self, device: str = "cpu"):
        super().__init__()
        # Use torchvision's pretrained weights
        resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        for p in resnet18.parameters():
            p.requires_grad = False
        # Remove final FC, keep up to avgpool
        self.backbone = nn.Sequential(
            resnet18.conv1,
            resnet18.bn1,
            resnet18.relu,
            resnet18.maxpool,
            resnet18.layer1,
            resnet18.layer2,
            resnet18.layer3,
            resnet18.layer4,
            resnet18.avgpool,  # yields (B, 512, 1, 1)
        )
        self.device = torch.device(device)
        self.eval().to(self.device)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        feats = self.backbone(x)           # (B, 512, 1, 1)
        feats = feats.view(feats.size(0), -1)  # (B, 512)
        return feats
