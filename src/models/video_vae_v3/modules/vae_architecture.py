"""Torch-native consolidated Video VAE architecture.

This module intentionally avoids relative imports, diffusers, memory/offload helpers,
and runtime-specific backends.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

class _Ops:
    Conv1d=nn.Conv1d
    Conv2d=nn.Conv2d
    Conv3d=nn.Conv3d
    Linear=nn.Linear
    LayerNorm=nn.LayerNorm
    GroupNorm=nn.GroupNorm
    Embedding=nn.Embedding

class MemoryState(Enum):
    DISABLED='disabled'
    ACTIVE='active'

@dataclass
class CausalEncoderOutput:
    latent_dist: object

@dataclass
class CausalDecoderOutput:
    sample: torch.Tensor

@dataclass
class CausalAutoencoderOutput:
    latent_dist: object

class AutoencoderKLOutput(tuple):
    def __new__(cls, latent_dist):
        return tuple.__new__(cls, (latent_dist,))
    latent_dist = property(lambda self: self[0])

class DecoderOutput(tuple):
    def __new__(cls, sample):
        return tuple.__new__(cls, (sample,))
    sample = property(lambda self: self[0])

class DiagonalGaussianDistribution:
    def __init__(self, moments: torch.Tensor):
        self.moments = moments
        self.mean, self.logvar = torch.chunk(moments, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5*self.logvar)
    def sample(self, generator=None):
        eps = torch.randn_like(self.mean)
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
        ops = operations or _Ops
        self.conv = ops.Conv3d(*args, **kwargs)
        self.inflation_mode = inflation_mode
        self.memory = None
    @property
    def weight(self):
        return self.conv.weight
    @property
    def bias(self):
        return self.conv.bias
    def forward(self, x: torch.Tensor, memory_state: MemoryState=MemoryState.DISABLED):
        return self.conv(x)

def init_causal_conv3d(*args, inflation_mode='tail', operations=None, **kwargs):
    return InflatedCausalConv3d(*args, inflation_mode=inflation_mode, operations=operations, **kwargs)

class Upsample3D(nn.Module):
    def __init__(self, channels: int, out_channels: int, temporal_up=False, spatial_up=True, operations=None, **kwargs):
        super().__init__()
        ops = operations or _Ops
        self.channels = channels
        self.out_channels = out_channels
        self.temporal_ratio = 2 if temporal_up else 1
        self.spatial_ratio = 2 if spatial_up else 1
        self.conv = init_causal_conv3d(channels, out_channels, 3, padding=1, operations=ops)
    def forward(self, x, output_size=None, memory_state: MemoryState=MemoryState.DISABLED, **kwargs):
        x = F.interpolate(x, scale_factor=(self.temporal_ratio,self.spatial_ratio,self.spatial_ratio), mode='nearest')
        return self.conv(x, memory_state=memory_state)

class Downsample3D(nn.Module):
    def __init__(self, channels: int, out_channels: int, temporal_down=False, spatial_down=True, operations=None, **kwargs):
        super().__init__()
        self.channels=channels
        self.out_channels=out_channels
        tr = 2 if temporal_down else 1
        sr = 2 if spatial_down else 1
        self.conv = init_causal_conv3d(channels, out_channels, kernel_size=(3 if temporal_down else 1,3 if spatial_down else 1,3 if spatial_down else 1), stride=(tr,sr,sr), padding=(1 if temporal_down else 0,1 if spatial_down else 0,1 if spatial_down else 0), operations=operations)
    def forward(self, x, memory_state: MemoryState=MemoryState.DISABLED, **kwargs):
        return self.conv(x, memory_state=memory_state)

class ResnetBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels=None, groups=32, operations=None, **kwargs):
        super().__init__()
        ops = operations or _Ops
        out_channels = out_channels or in_channels
        self.norm1 = ops.GroupNorm(groups, in_channels)
        self.conv1 = init_causal_conv3d(in_channels, out_channels, 3, padding=1, operations=ops)
        self.norm2 = ops.GroupNorm(groups, out_channels)
        self.conv2 = init_causal_conv3d(out_channels, out_channels, 3, padding=1, operations=ops)
        self.nin_shortcut = init_causal_conv3d(in_channels, out_channels, 1, operations=ops) if in_channels != out_channels else None
    def forward(self, x, temb=None, memory_state: MemoryState=MemoryState.DISABLED):
        h = self.conv1(F.silu(causal_norm_wrapper(self.norm1, x)), memory_state=memory_state)
        h = self.conv2(F.silu(causal_norm_wrapper(self.norm2, h)), memory_state=memory_state)
        x = self.nin_shortcut(x, memory_state=memory_state) if self.nin_shortcut is not None else x
        return x + h

class Encoder3D(nn.Module):
    def __init__(self, in_channels=3, out_channels=8, latent_channels=4, operations=None, **kwargs):
        super().__init__()
        ops=operations or _Ops
        self.conv_in = init_causal_conv3d(in_channels, out_channels, 3, padding=1, operations=ops)
        self.block = ResnetBlock3D(out_channels, out_channels, operations=ops)
        self.conv_out = init_causal_conv3d(out_channels, latent_channels*2, 3, padding=1, operations=ops)
    def forward(self, x, memory_state: MemoryState=MemoryState.DISABLED):
        x = self.conv_in(x, memory_state=memory_state)
        x = self.block(x, memory_state=memory_state)
        x = self.conv_out(x, memory_state=memory_state)
        return x

class Decoder3D(nn.Module):
    def __init__(self, latent_channels=4, out_channels=3, hidden_channels=8, operations=None, **kwargs):
        super().__init__()
        ops=operations or _Ops
        self.conv_in = init_causal_conv3d(latent_channels, hidden_channels, 3, padding=1, operations=ops)
        self.block = ResnetBlock3D(hidden_channels, hidden_channels, operations=ops)
        self.conv_out = init_causal_conv3d(hidden_channels, out_channels, 3, padding=1, operations=ops)
    def forward(self, z, memory_state: MemoryState=MemoryState.DISABLED):
        h = self.conv_in(z, memory_state=memory_state)
        h = self.block(h, memory_state=memory_state)
        return self.conv_out(h, memory_state=memory_state)

class VideoAutoencoderKL(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, latent_channels=4, operations=None, **kwargs):
        super().__init__()
        self.encoder = Encoder3D(in_channels=in_channels, latent_channels=latent_channels, operations=operations)
        self.decoder = Decoder3D(latent_channels=latent_channels, out_channels=out_channels, operations=operations)
    def encode(self, x, return_dict=True, **kwargs):
        moments = self.encoder(x)
        posterior = DiagonalGaussianDistribution(moments)
        return AutoencoderKLOutput(posterior) if return_dict else (posterior,)
    def decode(self, z, return_dict=True, **kwargs):
        dec = self.decoder(z)
        return DecoderOutput(dec) if return_dict else (dec,)
    def forward(self, sample, sample_posterior=False, return_dict=True, generator=None, **kwargs):
        posterior = self.encode(sample, return_dict=False)[0]
        z = posterior.sample(generator=generator) if sample_posterior else posterior.mode()
        dec = self.decode(z, return_dict=False)[0]
        return DecoderOutput(dec) if return_dict else (dec,)

class VideoAutoencoderKLWrapper(VideoAutoencoderKL):
    pass
