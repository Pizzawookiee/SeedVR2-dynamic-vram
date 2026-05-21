import torch
import os
import folder_paths
import comfy.ops
import comfy.model_management
import comfy.utils
from torch import nn

seedvr_model_path = os.path.join(folder_paths.models_dir, "seedvr2")
if not os.path.exists(seedvr_model_path):
    os.makedirs(seedvr_model_path, exist_ok=True)

folder_paths.add_model_folder_path("seedvr2", seedvr_model_path)

# Assuming the directory structure follows the YAML path definitions
# 3B Version uses models.dit_v2
try:
    from .src.models.dit_3b.nadit import NaDiT as NaDiT3B
except ImportError:
    NaDiT3B = None

# 7B Version uses models.dit
try:
    from .src.models.dit_7b.nadit import NaDiT as NaDiT7B
except ImportError:
    NaDiT7B = None

class SeedVR2ModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("seedvr2"),),
                "attention_mode": (["sdpa", "flash_attn_2", "flash_attn_3", "sageattn_2", "sageattn_3"], {"default": "sdpa"}),
                "weight_dtype": (["default", "fp16", "bf16", "fp8_e4m3fn"], {"default": "bf16"}),
            }
        }

    RETURN_TYPES = ("MODEL", "DIT_CONFIG")
    RETURN_NAMES = ("model", "config")
    FUNCTION = "load_model"
    CATEGORY = "SeedVR2"

    def load_model(self, ckpt_name, attention_mode, weight_dtype):
        ckpt_path = folder_paths.get_full_path("seedvr2", ckpt_name)
        
        # 1. Determine Model Version
        is_7b = "7b" in ckpt_name.lower()
        version_str = "7B" if is_7b else "3B"
        print(f"SeedVR2: Loading {version_str} configuration based on filename.")

        # 2. Set Dtype and Operations
        if weight_dtype == "fp16":
            target_dtype = torch.float16
        elif weight_dtype == "fp8_e4m3fn":
            target_dtype = torch.float8_e4m3fn
        else:
            target_dtype = torch.bfloat16
            
        ops = comfy.ops.manual_cast
        if weight_dtype == "fp8_e4m3fn":
            ops = comfy.ops.fp8_ops

        # 3. Resolve Parameters based on Version
        if is_7b:
            # 7B YAML Parameters
            num_layers = 36
            vid_dim = 3072
            model_class = NaDiT7B
            
            model_config = {
                "vid_in_channels": 33,
                "vid_out_channels": 16,
                "vid_dim": vid_dim,
                "txt_in_dim": 5120,
                "txt_dim": vid_dim,
                "emb_dim": 6 * vid_dim,
                "heads": 24,
                "head_dim": 128,
                "expand_ratio": 4,
                "norm": "fusedrms",
                "norm_eps": 1e-5,
                "ada": "single",
                "qk_bias": False,
                "qk_rope": True,
                "qk_norm": "fusedrms",
                "patch_size": (1, 2, 2),
                "num_layers": num_layers,
                "mm_layers": num_layers, # 7B usually processes all as MM
                "shared_mlp": False,
                "shared_qkv": False,
                "mlp_type": "normal",
                "block_type": ["mmdit_sr"] * num_layers,
                "window": [(4, 3, 3)] * num_layers,
                "window_method": ["720pwin_by_size_bysize", "720pswin_by_size_bysize"] * (num_layers // 2),
                "attention_mode": attention_mode,
            }
        else:
            # 3B YAML Parameters
            num_layers = 32
            vid_dim = 2560
            model_class = NaDiT3B
            
            model_config = {
                "vid_in_channels": 33,
                "vid_out_channels": 16,
                "vid_dim": vid_dim,
                "vid_out_norm": "fusedrms",
                "txt_in_dim": 5120,
                "txt_in_norm": "fusedln",
                "txt_dim": vid_dim,
                "emb_dim": 6 * vid_dim,
                "heads": 20,
                "head_dim": 128,
                "expand_ratio": 4,
                "norm": "fusedrms",
                "norm_eps": 1e-5,
                "ada": "single",
                "qk_bias": False,
                "qk_norm": "fusedrms",
                "patch_size": (1, 2, 2),
                "num_layers": num_layers,
                "mm_layers": 10,
                "mlp_type": "swiglu",
                "block_type": ["mmdit_sr"] * num_layers,
                "window": [(4, 3, 3)] * num_layers,
                "window_method": ["720pwin_by_size_bysize", "720pswin_by_size_bysize"] * (num_layers // 2),
                "rope_type": "mmrope3d",
                "rope_dim": 128,
                "attention_mode": attention_mode,
            }

        if model_class is None:
            raise ImportError(f"Could not find the NaDiT class for version {version_str}. Check your imports.")

        # 4. Instantiate on Meta Device
        model = model_class(
            **model_config,
            device="meta",
            dtype=target_dtype,
            operations=ops
        )

        # 5. Load Weights
        print(f"SeedVR2: Loading weights from {ckpt_name}...")
        sd = comfy.utils.load_torch_file(ckpt_path)
        
        # Materialize meta-tensors and load data
        model.load_state_dict(sd)

        # 6. Wrap for VRAM Management
        patcher = comfy.model_management.ModelPatcher(
            model, 
            load_device=comfy.model_management.get_torch_device(), 
            offload_device=comfy.model_management.intermediate_device(),
            current_device="cpu"
        )

        return (patcher, model_config)

NODE_CLASS_MAPPINGS = {
    "SeedVR2ModelLoader": SeedVR2ModelLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeedVR2ModelLoader": "SeedVR2 Model Loader"
}