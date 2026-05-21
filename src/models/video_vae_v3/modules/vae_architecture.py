"""Torch-native consolidated Video VAE architecture.

This module intentionally avoids relative imports, diffusers, memory/offload helpers,
and runtime-specific backends.
"""
from __future__ import annotations
from typing import Optional, Tuple, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiagonalGaussianDistribution:
    def __init__(self, moments: torch.Tensor):
        self.moments = moments
        self.mean, self.logvar = torch.chunk(moments, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5*self.logvar)
    def sample(self, generator=None):
        eps = torch.randn(
            self.mean.shape,
            generator=generator,
            device=self.mean.device,
            dtype=self.mean.dtype,
        )
        return self.mean + self.std * eps
    def mode(self):
        return self.mean

def remove_head(x: torch.Tensor) -> torch.Tensor:
    return x[:, :, 1:] if x.shape[2] > 1 else x

def causal_norm_wrapper(norm: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 5:
        b,c,t,h,w = x.shape
        y = x.permute(0,2,1,3,4).reshape(b*t,c,h,w)
        y = norm(y)
        return y.reshape(b,t,c,h,w).permute(0,2,1,3,4)
    return norm(x)

class InflatedCausalConv3d(nn.Module):
    def __init__(self, *args, inflation_mode: str='tail', operations=None, **kwargs):
        super().__init__()
        operations = operations or nn
        self.conv = operations.Conv3d(*args, **kwargs)
        self.inflation_mode = inflation_mode
    @property
    def weight(self):
        return self.conv.weight
    @property
    def bias(self):
        return self.conv.bias
    def forward(self, x: torch.Tensor):
        return self.conv(x)

def init_causal_conv3d(*args, inflation_mode='tail', operations=None, **kwargs):
    return InflatedCausalConv3d(*args, inflation_mode=inflation_mode, operations=operations, **kwargs)

class Upsample3D(nn.Module):
    def __init__(self, channels: int, out_channels: int, temporal_up=False, spatial_up=True, operations=None, **kwargs):
        super().__init__()
        operations = operations or nn
        self.channels = channels
        self.out_channels = out_channels
        self.temporal_ratio = 2 if temporal_up else 1
        self.spatial_ratio = 2 if spatial_up else 1
        self.conv = init_causal_conv3d(channels, out_channels, 3, padding=1, operations=operations)
    def forward(self, x, output_size=None, **kwargs):
        x = F.interpolate(x, scale_factor=(self.temporal_ratio,self.spatial_ratio,self.spatial_ratio), mode='nearest')
        return self.conv(x)

class Downsample3D(nn.Module):
    def __init__(self, channels: int, out_channels: int, temporal_down=False, spatial_down=True, operations=None, **kwargs):
        super().__init__()
        self.channels=channels
        self.out_channels=out_channels
        tr = 2 if temporal_down else 1
        sr = 2 if spatial_down else 1
        self.conv = init_causal_conv3d(channels, out_channels, kernel_size=(3 if temporal_down else 1,3 if spatial_down else 1,3 if spatial_down else 1), stride=(tr,sr,sr), padding=(1 if temporal_down else 0,1 if spatial_down else 0,1 if spatial_down else 0), operations=operations)
    def forward(self, x, **kwargs):
        return self.conv(x)


def _resolve_groups(num_channels: int, groups: int) -> int:
    groups = min(groups, num_channels)
    if num_channels % groups == 0:
        return groups
    return math.gcd(num_channels, groups) or 1

class ResnetBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels=None, groups=32, operations=None, act_fn: str = "silu", **kwargs):
        super().__init__()
        operations = operations or nn
        out_channels = out_channels or in_channels
        self.norm1 = operations.GroupNorm(_resolve_groups(in_channels, groups), in_channels)
        self.conv1 = init_causal_conv3d(in_channels, out_channels, 3, padding=1, operations=operations)
        self.norm2 = operations.GroupNorm(_resolve_groups(out_channels, groups), out_channels)
        self.conv2 = init_causal_conv3d(out_channels, out_channels, 3, padding=1, operations=operations)
        self.nin_shortcut = init_causal_conv3d(in_channels, out_channels, 1, operations=operations) if in_channels != out_channels else None
        self.act_fn = act_fn

    def _act(self, x):
        if self.act_fn == "silu":
            return F.silu(x)
        if self.act_fn == "relu":
            return F.relu(x)
        raise ValueError(f"Unsupported act_fn: {self.act_fn}")

    def forward(self, x, temb=None):
        h = self.conv1(self._act(causal_norm_wrapper(self.norm1, x)))
        h = self.conv2(self._act(causal_norm_wrapper(self.norm2, h)))
        x = self.nin_shortcut(x) if self.nin_shortcut is not None else x
        return x + h

class Encoder3D(nn.Module):
    def __init__(self, in_channels=3, block_out_channels=(8,), layers_per_block=1, latent_channels=4, operations=None, norm_num_groups=32, act_fn: str = "silu", inflation_mode: str = "tail", **kwargs):
        super().__init__()
        operations = operations or nn
        if len(block_out_channels) == 0:
            raise ValueError("block_out_channels must be non-empty")
        first = block_out_channels[0]
        self.conv_in = init_causal_conv3d(in_channels, first, 3, padding=1, operations=operations, inflation_mode=inflation_mode)
        blocks = []
        prev = first
        for ch in block_out_channels:
            for _ in range(layers_per_block):
                blocks.append(ResnetBlock3D(prev, ch, groups=norm_num_groups, operations=operations, act_fn=act_fn))
                prev = ch
        self.blocks = nn.ModuleList(blocks)
        self.conv_out = init_causal_conv3d(prev, latent_channels*2, 3, padding=1, operations=operations, inflation_mode=inflation_mode)

    def forward(self, x):
        x = self.conv_in(x)
        for block in self.blocks:
            x = block(x)
        x = self.conv_out(x)
        return x

class Decoder3D(nn.Module):
    def __init__(self, latent_channels=4, block_out_channels=(8,), layers_per_block=1, out_channels=3, operations=None, norm_num_groups=32, act_fn: str = "silu", inflation_mode: str = "tail", **kwargs):
        super().__init__()
        operations = operations or nn
        if len(block_out_channels) == 0:
            raise ValueError("block_out_channels must be non-empty")
        hidden_channels = block_out_channels[-1]
        self.conv_in = init_causal_conv3d(latent_channels, hidden_channels, 3, padding=1, operations=operations, inflation_mode=inflation_mode)
        blocks = []
        prev = hidden_channels
        for ch in reversed(block_out_channels):
            for _ in range(layers_per_block):
                blocks.append(ResnetBlock3D(prev, ch, groups=norm_num_groups, operations=operations, act_fn=act_fn))
                prev = ch
        self.blocks = nn.ModuleList(blocks)
        self.conv_out = init_causal_conv3d(prev, out_channels, 3, padding=1, operations=operations, inflation_mode=inflation_mode)

    def forward(self, z):
        h = self.conv_in(z)
        for block in self.blocks:
            h = block(h)
        return self.conv_out(h)

class VideoAutoencoderKL(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        block_out_channels=(8,),
        down_block_types=("DownEncoderBlock3D",),
        up_block_types=("UpDecoderBlock3D",),
        layers_per_block=1,
        latent_channels=4,
        norm_num_groups=32,
        act_fn="silu",
        inflation_mode="tail",
        operations=None,
        **kwargs,
    ):
        super().__init__()
        self.down_block_types = tuple(down_block_types)
        self.up_block_types = tuple(up_block_types)
        self.encoder = Encoder3D(
            in_channels=in_channels,
            block_out_channels=tuple(block_out_channels),
            layers_per_block=layers_per_block,
            latent_channels=latent_channels,
            operations=operations,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            inflation_mode=inflation_mode,
        )
        self.decoder = Decoder3D(
            latent_channels=latent_channels,
            block_out_channels=tuple(block_out_channels),
            layers_per_block=layers_per_block,
            out_channels=out_channels,
            operations=operations,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            inflation_mode=inflation_mode,
        )
    def encode(self, x, return_dict=True, **kwargs):
        moments = self.encoder(x)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior
    def decode(self, z, return_dict=True, **kwargs):
        dec = self.decoder(z)
        return dec
    def forward(self, sample, sample_posterior=False, return_dict=True, generator=None, **kwargs):
        posterior = self.encode(sample, return_dict=False)
        z = posterior.sample(generator=generator) if sample_posterior else posterior.mode()
        dec = self.decode(z, return_dict=False)
        return dec

class VideoAutoencoderKLWrapper(VideoAutoencoderKL):
    pass
