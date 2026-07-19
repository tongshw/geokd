import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import interpolate


def l2_norm(x):
    b, c, h, w = x.shape
    y = F.normalize(x.reshape(b, c * h * w))
    return y.reshape(b, c, h, w)


def round_channels(channels, divisor=8, min_value=4):
    channels = int(round(channels / divisor) * divisor)
    return max(min_value, channels)


class ResidualConvUnit(nn.Module):
    def __init__(self, features, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(features, features, kernel_size, padding=padding),
            nn.ReLU(True),
            nn.Conv2d(features, features, kernel_size, padding=padding),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.conv(x) + x


class FeatureFusionBlock(nn.Module):
    def __init__(self, features, kernel_size, with_skip=True):
        super().__init__()
        self.with_skip = with_skip
        if self.with_skip:
            self.resConfUnit1 = ResidualConvUnit(features, kernel_size)
        self.resConfUnit2 = ResidualConvUnit(features, kernel_size)

    def forward(self, x, skip_x=None):
        if skip_x is not None:
            x = self.resConfUnit1(x) + skip_x
        return self.resConfUnit2(x)


class SlimDPT(nn.Module):
    """Width-pruned DPT head used for the student localization model."""

    def __init__(
        self,
        input_dims,
        width_mult=0.5,
        hidden_dim=None,
        output_dims=None,
        kernel_size=3,
    ):
        super().__init__()
        assert len(input_dims) == 4
        if hidden_dim is None:
            hidden_dim = round_channels(512 * width_mult)
        if output_dims is None:
            output_dims = (
                round_channels(256 * width_mult),
                round_channels(64 * width_mult),
                round_channels(16 * width_mult, min_value=4),
            )

        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.output_dims = output_dims

        self.conv_0 = nn.Conv2d(input_dims[0], hidden_dim, 1, padding=0)
        self.conv_1 = nn.Conv2d(input_dims[1], hidden_dim, 1, padding=0)
        self.conv_2 = nn.Conv2d(input_dims[2], hidden_dim, 1, padding=0)
        self.conv_3 = nn.Conv2d(input_dims[3], hidden_dim, 1, padding=0)

        self.ref_0 = FeatureFusionBlock(hidden_dim, kernel_size)
        self.ref_1 = FeatureFusionBlock(hidden_dim, kernel_size)
        self.ref_2 = FeatureFusionBlock(hidden_dim, kernel_size)
        self.ref_3 = FeatureFusionBlock(hidden_dim, kernel_size, with_skip=False)

        self.out_conv_1 = nn.Sequential(
            nn.Conv2d(hidden_dim, max(hidden_dim // 2, output_dims[0]), 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(max(hidden_dim // 2, output_dims[0]), output_dims[0], 3, padding=1),
        )
        self.out_conv_2 = nn.Sequential(
            nn.Conv2d(hidden_dim, max(32, output_dims[1] * 2), 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(max(32, output_dims[1] * 2), output_dims[1], 3, padding=1),
        )
        self.out_conv_3 = nn.Sequential(
            nn.Conv2d(hidden_dim, max(16, output_dims[2] * 2), 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(max(16, output_dims[2] * 2), output_dims[2], 3, padding=1),
        )

    def forward(self, feats):
        assert len(feats) == 4
        feats = list(feats)
        feats[0] = self.conv_0(feats[0])
        feats[1] = self.conv_1(feats[1])
        feats[2] = self.conv_2(feats[2])
        feats[3] = self.conv_3(feats[3])

        feats = [interpolate(x, scale_factor=2) for x in feats]

        out = self.ref_3(feats[3], None)
        out1 = self.ref_2(feats[2], out)
        out2 = self.ref_1(feats[1], out1)
        out3 = self.ref_0(feats[0], out2)

        out1 = self.out_conv_1(out1)
        out2 = self.out_conv_2(interpolate(out2, scale_factor=2))
        out3 = self.out_conv_3(interpolate(out3, scale_factor=4))
        return [l2_norm(out1), l2_norm(out2), l2_norm(out3)]
