import torch
import torch.nn as nn
import os
import folder_paths
import comfy.ops
import comfy.model_management
import comfy.utils
import logging
from typing import Dict, Any
import math


seedvr_model_path = os.path.join(folder_paths.models_dir, "seedvr2")
if not os.path.exists(seedvr_model_path):
    os.makedirs(seedvr_model_path, exist_ok=True)

folder_paths.add_model_folder_path("seedvr2", seedvr_model_path)

# Import the wrapper class
try:
    from .src.models.video_vae_v3.modules.attn_video_vae import VideoAutoencoderKLWrapper
    from .src.models.video_vae_v3.modules.causal_inflation_lib import InflatedCausalConv3d
except ImportError:
    # Adjust path based on your actual local file structure
    VideoAutoencoderKLWrapper = None


def swap_layers_recursively(model, target_ops):
    """
    Recursively replaces standard torch.nn modules with custom operations 
    from the provided target_ops class (e.g., manual_cast, fp8_ops).
    """
    for name, child in model.named_children():
        new_layer = None
        
        # 1. Handle Linear Layers
        if isinstance(child, nn.Linear) and hasattr(target_ops, "Linear"):
            new_layer = target_ops.Linear(
                child.in_features, 
                child.out_features, 
                bias=child.bias is not None,
                device=child.weight.device,
                dtype=child.weight.dtype
            )
            
        # 2. Handle Convolutional Layers (1D, 2D)
        elif isinstance(child, (nn.Conv1d, nn.Conv2d)):
            dim = 2 if isinstance(child, nn.Conv2d) else 1
            op_name = f"Conv{dim}d"
            
            if hasattr(target_ops, op_name):
                target_cls = getattr(target_ops, op_name)
                new_layer = target_cls(
                    child.in_channels,
                    child.out_channels,
                    child.kernel_size,
                    stride=child.stride,
                    padding=child.padding,
                    dilation=child.dilation,
                    groups=child.groups,
                    bias=child.bias is not None,
                    padding_mode=child.padding_mode,
                    device=child.weight.device,
                    dtype=child.weight.dtype
                )

        # 2b. Handle InflatedCausalConv3d
        elif isinstance(child, InflatedCausalConv3d) and hasattr(target_ops, "Conv3d"):

            class PatchedCausalConv3d(InflatedCausalConv3d, target_ops.Conv3d):
                def __init__(self, layer):
                    # 1. Initialize target_ops.Conv3d using the exact parameters from the old layer
                    super().__init__(
                        in_channels=layer.in_channels,
                        out_channels=layer.out_channels,
                        kernel_size=layer.kernel_size,
                        stride=layer.stride,
                        padding=(layer.temporal_padding, *layer.padding[1:]), # Restore true padding for init
                        dilation=layer.dilation,
                        groups=layer.groups,
                        bias=(layer.bias is not None),
                        device=layer.weight.device,
                        dtype=layer.weight.dtype,
                        inflation_mode = layer.inflation_mode,
                        memory_device = layer.memory_device
                    )

            # Instantiate, share weight/bias references, and replace
            new_layer = PatchedCausalConv3d(child)
            new_layer.weight = child.weight
            if child.bias is not None:
                new_layer.bias = child.bias

            setattr(model, name, new_layer)

        # 3. Handle Normalization Layers
        elif isinstance(child, nn.LayerNorm) and hasattr(target_ops, "LayerNorm"):
            new_layer = target_ops.LayerNorm(
                child.normalized_shape,
                eps=child.eps,
                elementwise_affine=child.elementwise_affine,
                device=child.weight.device if child.elementwise_affine else None,
                dtype=child.weight.dtype if child.elementwise_affine else None
            )
            
        elif isinstance(child, nn.GroupNorm) and hasattr(target_ops, "GroupNorm"):
            new_layer = target_ops.GroupNorm(
                child.num_groups,
                child.num_channels,
                eps=child.eps,
                affine=child.affine,
                device=child.weight.device if child.affine else None,
                dtype=child.weight.dtype if child.affine else None
            )

        # 4. Handle Embeddings
        elif isinstance(child, nn.Embedding) and hasattr(target_ops, "Embedding"):
            new_layer = target_ops.Embedding(
                child.num_embeddings,
                child.embedding_dim,
                padding_idx=child.padding_idx,
                max_norm=child.max_norm,
                norm_type=child.norm_type,
                scale_grad_by_freq=child.scale_grad_by_freq,
                sparse=child.sparse,
                device=child.weight.device,
                dtype=child.weight.dtype
            )

        # 5. Apply replacement or recurse
        if new_layer is not None:
            # Transfer existing weights/bias to the new layer to avoid losing them 
            # if they were already initialized or partially loaded.
            with torch.no_grad():
                if child.weight is not None and new_layer.weight is not None:
                    new_layer.weight.copy_(child.weight)
                if child.bias is not None and new_layer.bias is not None:
                    new_layer.bias.copy_(child.bias)
            
            setattr(model, name, new_layer)
        else:
            # If no replacement happened, recurse into children
            swap_layers_recursively(child, target_ops)

    return model

class SeedVR2VAELoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae_name": (folder_paths.get_filename_list("seedvr2"),),
                "weight_dtype": (["default", "fp16", "bf16"], {"default": "bf16"}),
            }
        }

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "load_vae"
    CATEGORY = "SeedVR2"

    def load_vae(self, vae_name, weight_dtype):
        vae_path = folder_paths.get_full_path("seedvr2", vae_name)
        
        target_dtype = torch.float16 if weight_dtype == "fp16" else torch.bfloat16

        # corrected to match your class __init__ EXACTLY
        vae_config = {
            "in_channels": 3,
            "out_channels": 3,
            "block_out_channels": (128, 256, 512, 512), # Cast to Tuple
            "down_block_types": ("DownEncoderBlock3D", "DownEncoderBlock3D", "DownEncoderBlock3D", "DownEncoderBlock3D"), # Cast to Tuple
            "up_block_types": ("UpDecoderBlock3D", "UpDecoderBlock3D", "UpDecoderBlock3D", "UpDecoderBlock3D"), # Cast to Tuple
            "layers_per_block": 2,
            "latent_channels": 16,
            "norm_num_groups": 32,
            "act_fn": "silu",
            "temporal_scale_num": 2,
            "inflation_mode": "tail", # Changed from 'pad' to 'tail' to match standard _inflation_mode_t
            "slicing_sample_min_size": 4,
            "use_quant_conv": False,
            "use_post_quant_conv": False,
            "attention": True, # Added explicitly from signature
            "spatial_downsample_factor": 8,
            "temporal_downsample_factor": 4,
            "freeze_encoder": False,
        }

        # Instantiate
        # This matches: VideoAutoencoderKL(..., device=None, dtype=None, operations=...)
        model = VideoAutoencoderKLWrapper(
            **vae_config
        )

        # Loading logic
        default_ops = comfy.ops.manual_cast

        swap_layers_recursively(model, default_ops)

        sd = comfy.utils.load_torch_file(vae_path)

        model.set_causal_slicing(split_size=4, memory_device="same")
        vae = comfy.sd.VAE(sd={}, config=None)
        vae.latent_dim = 3

        vae.first_stage_model = model.eval()

        # Manually set the attributes that the early return skipped
        vae.vae_dtype = target_dtype
        vae.first_stage_model.to(vae.vae_dtype)
        comfy.model_management.archive_model_dtypes(vae.first_stage_model)
        vae.working_dtypes = [torch.bfloat16, torch.float32, torch.float16]

        vae.upscale_ratio = (lambda a: max(0, a * 4 - 3), 16, 16)
        vae.downscale_ratio = (lambda a: max(0, math.floor((a + 3) / 4)), 16, 16)
        vae.upscale_index_formula = (4, 16, 16)
        vae.downscale_index_formula = (4, 16, 16)
        vae.latent_channels = 16 # Matches your latent_channels: 16 in config

        device = comfy.model_management.vae_device()
        offload_device = comfy.model_management.vae_offload_device()
        vae.device = device
        
        vae.output_device = comfy.model_management.intermediate_device()
        mp = comfy.model_patcher.CoreModelPatcher
        if vae.disable_offload:
            mp = comfy.model_patcher.ModelPatcher
        vae.patcher = mp(vae.first_stage_model, load_device=vae.device, offload_device=offload_device)
        vae.first_stage_model.load_state_dict(sd, assign=vae.patcher.is_dynamic())
        vae.size = comfy.model_management.module_size(vae.first_stage_model)
        vae.throw_exception_if_invalid()

        return (vae,)


# Node Mapping for ComfyUI
NODE_CLASS_MAPPINGS = {
    "SeedVR2VAELoader": SeedVR2VAELoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeedVR2VAELoader": "SeedVR2 VAE Loader"
}