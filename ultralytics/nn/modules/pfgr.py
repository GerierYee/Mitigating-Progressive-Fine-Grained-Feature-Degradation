# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Modules used by the PFGR realization in CMD-YOLO26.

Canonical paper names are exposed together with the historical class names
used by the original experiment checkpoints. The aliases keep old YAML files
and state dictionaries loadable without changing their behavior.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .conv import Conv

__all__ = (
    "CAE_Module",
    "FGBSCA",
    "FLPSPD",
    "LHMSP",
    "LightASFF",
    "MHP_Block",
    "P2ASFF",
    "SPD_Downsample",
    "SimAM",
)


class SimAM(nn.Module):
    """Parameter-free spatial energy modulation used inside FG-BSCA."""

    def __init__(self, e_lambda: float = 1e-4):
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        count = max(height * width - 1, 1)
        mean = x.mean(dim=(2, 3), keepdim=True)
        variance = ((x - mean) ** 2).sum(dim=(2, 3), keepdim=True) / count
        energy = (x - mean) ** 2 / (4 * (variance + self.e_lambda)) + 0.5
        return x * torch.sigmoid(energy)


class FLPSPD(nn.Module):
    """Fine-grained local-preserving space-to-depth downsampling."""

    def __init__(self, c1: int, c2: int, scale: int = 2):
        super().__init__()
        self.unshuffle = nn.PixelUnshuffle(scale)
        self.conv = nn.Conv2d(c1 * scale**2, c2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(self.unshuffle(x))))


class LHMSP(nn.Module):
    """Lightweight hierarchical multi-scale perception block."""

    def __init__(self, c1: int, c2: int):
        super().__init__()
        if c2 % 4:
            raise ValueError(f"LHMSP requires c2 divisible by 4, received {c2}")
        branch_channels = c2 // 4
        self.cv0 = nn.Conv2d(c1, branch_channels, 3, 1, 1, bias=False)
        self.cv1 = nn.Conv2d(c1, branch_channels, 3, 1, 2, dilation=2, bias=False)
        self.cv2 = nn.Conv2d(c1, branch_channels, 3, 1, 3, dilation=3, bias=False)
        self.cv3 = nn.Conv2d(c1, branch_channels, 3, 1, 5, dilation=5, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
        self.fuse = nn.Conv2d(c2, c2, 1, bias=False)
        self.skip = nn.Identity() if c1 == c2 else nn.Conv2d(c1, c2, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        branches = torch.cat((self.cv0(x), self.cv1(x), self.cv2(x), self.cv3(x)), dim=1)
        return residual + self.fuse(self.act(self.bn(branches)))


class FGBSCA(nn.Module):
    """Fine-grained background-suppressed coordinate attention."""

    def __init__(self, c1: int, c2: int | None = None, ratio: int = 16):
        super().__init__()
        c2 = c2 or c1
        hidden = max(8, c1 // ratio)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(c1, hidden, 1, bias=True)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.act = nn.SiLU()
        self.conv_h = nn.Conv2d(hidden, c2, 1, bias=True)
        self.conv_w = nn.Conv2d(hidden, c2, 1, bias=True)
        self.simam = SimAM()
        self.skip = nn.Identity() if c1 == c2 else nn.Conv2d(c1, c2, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        _, _, height, width = x.shape
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        fused = self.act(self.bn1(self.conv1(torch.cat((x_h, x_w), dim=2))))
        x_h, x_w = torch.split(fused, (height, width), dim=2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w.permute(0, 1, 3, 2)).sigmoid()
        return self.simam(residual * a_h * a_w)


class P2ASFF(nn.Module):
    """Lightweight adaptive fusion of P2, upsampled P3, and upsampled P4.

    The preceding Concat is expected to contain ``[P2, P3_up, P4_up]`` with
    channel proportions ``1:2:4``. This is equivalent to the historical
    LightASFF implementation for the nano scale without hard-coded channels.
    """

    def __init__(self, c1: int, c2: int):
        super().__init__()
        expected = 7 * c2
        if c1 != expected:
            raise ValueError(f"P2ASFF expects concatenated channels 7*c2={expected}, received {c1}")
        self.c2 = c2
        self.conv_p3 = Conv(2 * c2, c2, 1, act=False)
        self.conv_p4 = Conv(4 * c2, c2, 1, act=False)
        self.weight_conv = nn.Conv2d(3 * c2, 3, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p2, p3, p4 = torch.split(x, (self.c2, 2 * self.c2, 4 * self.c2), dim=1)
        p3 = self.conv_p3(p3)
        p4 = self.conv_p4(p4)
        weights = torch.softmax(self.weight_conv(torch.cat((p2, p3, p4), dim=1)), dim=1)
        return weights[:, 0:1] * p2 + weights[:, 1:2] * p3 + weights[:, 2:3] * p4


# Historical names retained for the released experiment YAMLs/checkpoints.
SPD_Downsample = FLPSPD
MHP_Block = LHMSP
CAE_Module = FGBSCA
LightASFF = P2ASFF
