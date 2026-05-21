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
    from .src.models.video_vae_v3.modules.vae_architecture import VideoAutoencoderKLWrapper
except ImportError:
    # Adjust path based on your actual local file structure
    VideoAutoencoderKLWrapper = None



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

        default_ops = comfy.ops.manual_cast

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
            "operations": default_ops
        }

        # Instantiate
        # This matches: VideoAutoencoderKL(..., device=None, dtype=None, operations=...)
        model = VideoAutoencoderKLWrapper(
            **vae_config
        )


        sd = comfy.utils.load_torch_file(vae_path)

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