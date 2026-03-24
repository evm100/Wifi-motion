"""Domain-adaptive CSI classification with gradient reversal."""

from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models as models


class _GradientReversalFn(torch.autograd.Function):
    """Negates gradients during backward pass."""

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """Gradient Reversal Layer — identity in forward, negates gradients in backward."""

    def __init__(self, lambda_: float = 1.0) -> None:
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GradientReversalFn.apply(x, self.lambda_)


class DomainAdaptiveCSINet(nn.Module):
    """
    Adversarial domain adaptation for cross-room CSI classification.

    Shared ResNet feature extractor with:
    - activity_classifier head (main task)
    - domain_discriminator head (with GRL, adversarial)

    Input: [B, C, H, W] spectrogram
    Output: (activity_logits [B, n_classes], domain_logits [B, n_domains])
    """

    def __init__(
        self,
        n_input_channels: int = 30,
        n_classes: int = 7,
        n_domains: int = 3,
        lambda_domain: float = 1.0,
    ) -> None:
        super().__init__()

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, 7, stride=2, padding=3, bias=False),
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
            nn.Flatten(),
        )

        # Initialize first conv by repeating pretrained weights
        with torch.no_grad():
            pretrained_weight = resnet.conv1.weight.data
            repeats = n_input_channels // 3 + 1
            expanded = pretrained_weight.repeat(1, repeats, 1, 1)
            self.feature_extractor[0].weight.data = expanded[:, :n_input_channels, :, :]

        self.activity_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes),
        )

        self.domain_discriminator = nn.Sequential(
            GradientReversalLayer(lambda_=lambda_domain),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_domains),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(x)
        activity_pred = self.activity_classifier(features)
        domain_pred = self.domain_discriminator(features)
        return activity_pred, domain_pred
