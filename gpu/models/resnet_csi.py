"""ResNet18 adapted for multi-channel CSI spectrogram classification."""

import torch
import torch.nn as nn
import torchvision.models as models


class CSIResNet(nn.Module):
    """
    ResNet18 adapted for multi-channel CSI spectrogram classification.

    Input: [B, C, H, W] where C = n_nodes * n_pcs_per_node (default 30)
    Output: [B, n_classes] logits
    """

    def __init__(
        self,
        n_input_channels: int = 30,
        n_classes: int = 7,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)

        # Replace first conv to accept n_input_channels instead of 3
        self.conv1 = nn.Conv2d(
            n_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Initialize by repeating pretrained 3-channel weights
        if pretrained:
            with torch.no_grad():
                pretrained_weight = resnet.conv1.weight.data  # [64, 3, 7, 7]
                repeats = n_input_channels // 3 + 1
                expanded = pretrained_weight.repeat(1, repeats, 1, 1)
                self.conv1.weight.data = expanded[:, :n_input_channels, :, :]

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
