"""
AdaFace IR-101 Backbone

Vendored from https://github.com/mk-minchul/AdaFace (MIT License).
This is the IResNet-101 backbone used by AdaFace for face recognition.
Architecture matches the original paper's IR-101 variant.

Input:  (B, 3, 112, 112) — aligned face images, pixel range [-1, 1]
Output: (B, 512) — L2-normalized face embeddings
"""

import torch
import torch.nn as nn
from typing import List


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for AdaFace."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.avg_pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w.expand_as(x)


class IBasicBlock(nn.Module):
    """
    Improved ResNet basic block for AdaFace IR-101.
    BN -> Conv -> BN -> PReLU -> Conv -> BN -> SE -> residual add
    """
    expansion = 1

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        downsample: nn.Module = None,
        use_se: bool = True,
    ):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch, eps=1e-5)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch, eps=1e-5)
        self.prelu = nn.PReLU(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch, eps=1e-5)
        self.se = SEBlock(out_ch) if use_se else nn.Identity()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return out + identity


class IResBackbone(nn.Module):
    """
    IResNet backbone used by AdaFace.
    For IR-101: layers = [3, 13, 30, 3], same as ArcFace iresnet100
    but with different head normalization (AdaFace-specific).
    """

    def __init__(
        self,
        layers: List[int],
        embedding_dim: int = 512,
        dropout: float = 0.0,
        use_se: bool = True,
    ):
        super().__init__()
        self.in_channels = 64
        self.use_se = use_se

        # Stem
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-5)
        self.prelu = nn.PReLU(64)

        # Residual stages
        self.layer1 = self._make_layer(64, layers[0], stride=2)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        # Head
        self.bn2 = nn.BatchNorm2d(512, eps=1e-5)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(512 * 7 * 7, embedding_dim)
        self.bn3 = nn.BatchNorm1d(embedding_dim, eps=1e-5, affine=False)

        self._init_weights()

    def _make_layer(self, out_ch: int, num_blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_ch:
            if self.in_channels == out_ch:
                # Same channels, just spatial downsample — use parameter-free MaxPool
                # (matches CVLFace/insightface architecture)
                downsample = nn.MaxPool2d(1, stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(self.in_channels, out_ch, 1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_ch, eps=1e-5),
                )
        layers = [IBasicBlock(self.in_channels, out_ch, stride, downsample, self.use_se)]
        self.in_channels = out_ch
        for _ in range(1, num_blocks):
            layers.append(IBasicBlock(out_ch, out_ch, use_se=self.use_se))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 112, 112) in [-1, 1] range (AdaFace normalization)
        Returns:
            (B, 512) embeddings (not L2-normalized — caller should normalize)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn2(x)
        x = self.dropout(x)
        x = x.flatten(1)
        x = self.fc(x)
        x = self.bn3(x)
        return x


def build_adaface_ir101(embedding_dim: int = 512, use_se: bool = True) -> IResBackbone:
    """Factory for AdaFace IR-101 backbone (layers = [3, 13, 30, 3])."""
    return IResBackbone(
        layers=[3, 13, 30, 3],
        embedding_dim=embedding_dim,
        dropout=0.4,
        use_se=use_se,
    )
