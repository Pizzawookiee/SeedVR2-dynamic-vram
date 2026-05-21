from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiagonalGaussianDistribution:
    def __init__(self, moments: torch.Tensor):
        self.moments = moments
        self.mean, self.logvar = torch.chunk(moments, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)

    def sample(self, generator=None):
        eps = torch.randn(self.mean.shape, generator=generator, device=self.mean.device, dtype=self.mean.dtype)
        return self.mean + self.std * eps

    def mode(self):
        return self.mean

    @property
    def shape(self):
        return self.mean.shape

    def to(self, *args, **kwargs):
        self.moments = self.moments.to(*args, **kwargs)
        self.mean = self.mean.to(*args, **kwargs)
        self.logvar = self.logvar.to(*args, **kwargs)
        self.std = self.std.to(*args, **kwargs)
        return self


def _resolve_groups(num_channels: int, groups: int) -> int:
    groups = min(groups, num_channels)
    if num_channels % groups == 0:
        return groups
    return math.gcd(num_channels, groups) or 1


def causal_norm_wrapper(norm: nn.Module, x: torch.Tensor) -> torch.Tensor:
    b, c, t, h, w = x.shape
    y = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    y = norm(y)
    return y.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)


_INFLATED_CONV_CLASS_CACHE = {}


def _inflated_conv_class(base_conv3d):
    cached = _INFLATED_CONV_CLASS_CACHE.get(base_conv3d)
    if cached is not None:
        return cached

    class InflatedCausalConv3d(base_conv3d):
        def __init__(self, *args, **kwargs):
            padding = kwargs.pop("padding", 0)
            if isinstance(padding, int):
                padding = (padding, padding, padding)
            self.temporal_padding = int(padding[0]) * 2
            super().__init__(*args, padding=(0, int(padding[1]), int(padding[2])), **kwargs)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.temporal_padding > 0:
                head = x[:, :, :1].repeat(1, 1, self.temporal_padding, 1, 1)
                x = torch.cat([head, x], dim=2)
            return super().forward(x)

    _INFLATED_CONV_CLASS_CACHE[base_conv3d] = InflatedCausalConv3d
    return InflatedCausalConv3d


class InflatedCausalConv3d(_inflated_conv_class(nn.Conv3d)):
    """Default torch Conv3d-based causal conv class."""


def init_causal_conv3d(*args, operations=None, **kwargs):
    operations = operations or nn
    base_conv3d = operations.Conv3d
    conv_cls = _inflated_conv_class(base_conv3d)
    return conv_cls(*args, **kwargs)


class Upsample3D(nn.Module):
    def __init__(self, channels: int, temporal_up: bool = True, operations=None):
        super().__init__()
        self.conv = init_causal_conv3d(channels, channels, 3, padding=1, operations=operations)
        self.temporal_up = temporal_up
        scale_mul = 8 if temporal_up else 4
        self.upscale_conv = init_causal_conv3d(channels, channels * scale_mul, 1, padding=0, operations=operations)

    def forward(self, x):
        x = self.conv(x)
        x = self.upscale_conv(x)
        b, c, t, h, w = x.shape
        if self.temporal_up:
            x = x.view(b, c // 8, 2, 2, 2, t, h, w).permute(0, 1, 5, 2, 6, 3, 7, 4).reshape(b, c // 8, t * 2, h * 2, w * 2)
        else:
            x = x.view(b, c // 4, 2, 2, t, h, w).permute(0, 1, 4, 2, 5, 3, 6).reshape(b, c // 4, t, h * 2, w * 2)
        return x


class Downsample3D(nn.Module):
    def __init__(self, channels: int, first: bool = False, temporal_down: bool = False, operations=None):
        super().__init__()
        if temporal_down:
            kernel = (3, 3, 3)
            padding = (1, 1, 1)
            stride = (2, 2, 2)
        else:
            kernel = (1, 3, 3) if first else (3, 3, 3)
            padding = (0, 1, 1) if first else (1, 1, 1)
            stride = (1, 2, 2)
        self.conv = init_causal_conv3d(channels, channels, kernel_size=kernel, stride=stride, padding=padding, operations=operations)

    def forward(self, x):
        return self.conv(x)


class ResnetBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, groups=32, act_fn="silu", operations=None):
        super().__init__()
        operations = operations or nn
        self.norm1 = operations.GroupNorm(_resolve_groups(in_channels, groups), in_channels)
        self.conv1 = init_causal_conv3d(in_channels, out_channels, 3, padding=1, operations=operations)
        self.norm2 = operations.GroupNorm(_resolve_groups(out_channels, groups), out_channels)
        self.conv2 = init_causal_conv3d(out_channels, out_channels, 3, padding=1, operations=operations)
        self.conv_shortcut = init_causal_conv3d(in_channels, out_channels, 1, padding=0, operations=operations) if in_channels != out_channels else None
        self.act_fn = act_fn

    def _act(self, x):
        return F.silu(x) if self.act_fn == "silu" else F.relu(x)

    def forward(self, x):
        h = self.conv1(self._act(causal_norm_wrapper(self.norm1, x)))
        h = self.conv2(self._act(causal_norm_wrapper(self.norm2, h)))
        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)
        return x + h


class Attention3D(nn.Module):
    def __init__(self, channels, groups=32, operations=None):
        super().__init__()
        operations = operations or nn
        self.group_norm = operations.GroupNorm(_resolve_groups(channels, groups), channels)
        self.to_q = operations.Linear(channels, channels)
        self.to_k = operations.Linear(channels, channels)
        self.to_v = operations.Linear(channels, channels)
        self.to_out = nn.ModuleList([operations.Linear(channels, channels)])

    def forward(self, x):
        b, c, t, h, w = x.shape
        y = causal_norm_wrapper(self.group_norm, x).permute(0, 2, 3, 4, 1).reshape(b, t * h * w, c)
        q, k, v = self.to_q(y), self.to_k(y), self.to_v(y)
        a = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / math.sqrt(c), dim=-1)
        y = self.to_out[0](torch.bmm(a, v)).reshape(b, t, h, w, c).permute(0, 4, 1, 2, 3)
        return x + y


class UNetMidBlock3D(nn.Module):
    def __init__(self, channels, groups=32, operations=None):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResnetBlock3D(channels, channels, groups=groups, operations=operations),
            ResnetBlock3D(channels, channels, groups=groups, operations=operations),
        ])
        self.attentions = nn.ModuleList([Attention3D(channels, groups=groups, operations=operations)])

    def forward(self, x):
        x = self.resnets[0](x)
        x = self.attentions[0](x)
        x = self.resnets[1](x)
        return x


class DownEncoderBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, add_downsample=True, first=False, temporal_down=False, groups=32, operations=None):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResnetBlock3D(in_channels, out_channels, groups=groups, operations=operations),
            ResnetBlock3D(out_channels, out_channels, groups=groups, operations=operations),
        ])
        self.downsamplers = nn.ModuleList([Downsample3D(out_channels, first=first, temporal_down=temporal_down, operations=operations)]) if add_downsample else nn.ModuleList([])

    def forward(self, x):
        for r in self.resnets:
            x = r(x)
        for d in self.downsamplers:
            x = d(x)
        return x


class UpDecoderBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, add_upsample=True, temporal_up=True, groups=32, operations=None):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResnetBlock3D(in_channels, out_channels, groups=groups, operations=operations),
            ResnetBlock3D(out_channels, out_channels, groups=groups, operations=operations),
            ResnetBlock3D(out_channels, out_channels, groups=groups, operations=operations),
        ])
        self.upsamplers = nn.ModuleList([Upsample3D(out_channels, temporal_up=temporal_up, operations=operations)]) if add_upsample else nn.ModuleList([])

    def forward(self, x):
        for r in self.resnets:
            x = r(x)
        for u in self.upsamplers:
            x = u(x)
        return x


class Encoder3D(nn.Module):
    def __init__(self, in_channels=3, block_out_channels=(128, 256, 512, 512), latent_channels=16, norm_num_groups=32, temporal_scale_num=2, operations=None):
        super().__init__()
        operations = operations or nn
        self.conv_in = init_causal_conv3d(in_channels, block_out_channels[0], 3, padding=1, operations=operations)
        self.down_blocks = nn.ModuleList([
            DownEncoderBlock3D(block_out_channels[0], block_out_channels[0], add_downsample=True, first=True, temporal_down=(0 < temporal_scale_num), groups=norm_num_groups, operations=operations),
            DownEncoderBlock3D(block_out_channels[0], block_out_channels[1], add_downsample=True, temporal_down=(1 < temporal_scale_num), groups=norm_num_groups, operations=operations),
            DownEncoderBlock3D(block_out_channels[1], block_out_channels[2], add_downsample=True, temporal_down=(2 < temporal_scale_num), groups=norm_num_groups, operations=operations),
            DownEncoderBlock3D(block_out_channels[2], block_out_channels[3], add_downsample=False, groups=norm_num_groups, operations=operations),
        ])
        self.mid_block = UNetMidBlock3D(block_out_channels[-1], groups=norm_num_groups, operations=operations)
        self.conv_norm_out = operations.GroupNorm(_resolve_groups(block_out_channels[-1], norm_num_groups), block_out_channels[-1])
        self.conv_out = init_causal_conv3d(block_out_channels[-1], latent_channels * 2, 3, padding=1, operations=operations)

    def forward(self, x):
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x)
        x = self.mid_block(x)
        x = F.silu(causal_norm_wrapper(self.conv_norm_out, x))
        return self.conv_out(x)


class Decoder3D(nn.Module):
    def __init__(self, out_channels=3, block_out_channels=(128, 256, 512, 512), latent_channels=16, norm_num_groups=32, temporal_scale_num=2, operations=None):
        super().__init__()
        operations = operations or nn
        self.conv_in = init_causal_conv3d(latent_channels, block_out_channels[-1], 3, padding=1, operations=operations)
        self.mid_block = UNetMidBlock3D(block_out_channels[-1], groups=norm_num_groups, operations=operations)
        rev = list(reversed(block_out_channels))
        self.up_blocks = nn.ModuleList([
            UpDecoderBlock3D(rev[0], rev[0], add_upsample=True, temporal_up=(0 < temporal_scale_num), groups=norm_num_groups, operations=operations),
            UpDecoderBlock3D(rev[0], rev[1], add_upsample=True, temporal_up=(1 < temporal_scale_num), groups=norm_num_groups, operations=operations),
            UpDecoderBlock3D(rev[1], rev[2], add_upsample=True, temporal_up=(2 < temporal_scale_num), groups=norm_num_groups, operations=operations),
            UpDecoderBlock3D(rev[2], rev[3], add_upsample=False, groups=norm_num_groups, operations=operations),
        ])
        self.conv_norm_out = operations.GroupNorm(_resolve_groups(block_out_channels[0], norm_num_groups), block_out_channels[0])
        self.conv_out = init_causal_conv3d(block_out_channels[0], out_channels, 3, padding=1, operations=operations)

    def forward(self, z):
        z = self.conv_in(z)
        z = self.mid_block(z)
        for block in self.up_blocks:
            z = block(z)
        z = F.silu(causal_norm_wrapper(self.conv_norm_out, z))
        return self.conv_out(z)


class VideoAutoencoderKL(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, block_out_channels=(128, 256, 512, 512), latent_channels=16, norm_num_groups=32, temporal_scale_num=2, operations=None, **kwargs):
        super().__init__()
        self.encoder = Encoder3D(in_channels=in_channels, block_out_channels=block_out_channels, latent_channels=latent_channels, norm_num_groups=norm_num_groups, temporal_scale_num=temporal_scale_num, operations=operations)
        self.decoder = Decoder3D(out_channels=out_channels, block_out_channels=block_out_channels, latent_channels=latent_channels, norm_num_groups=norm_num_groups, temporal_scale_num=temporal_scale_num, operations=operations)

    def encode_distribution(self, x, **kwargs):
        return DiagonalGaussianDistribution(self.encoder(x))

    def encode(self, x, sample_posterior=False, generator=None, **kwargs):
        posterior = self.encode_distribution(x, **kwargs)
        return posterior.sample(generator=generator) if sample_posterior else posterior.mode()

    def decode(self, z, **kwargs):
        return self.decoder(z)

    def forward(self, sample, sample_posterior=False, generator=None, **kwargs):
        z = self.encode(sample, sample_posterior=sample_posterior, generator=generator, **kwargs)
        return self.decode(z)


class VideoAutoencoderKLWrapper(VideoAutoencoderKL):
    pass
