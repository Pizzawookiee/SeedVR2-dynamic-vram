# SeedVR2 Custom Nodes for ComfyUI
# Implementation for NaDiT and Video VAE v3

from .model_loader import SeedVR2ModelLoader
from .vae_loader import SeedVR2VAELoader

# This dictionary maps the internal name used in workflow JSONs 
# to the actual Python class.
NODE_CLASS_MAPPINGS = {
    "SeedVR2ModelLoader": SeedVR2ModelLoader,
    "SeedVR2VAELoader": SeedVR2VAELoader
}

# This dictionary defines how the nodes appear in the 
# ComfyUI "Add Node" menu.
NODE_DISPLAY_NAME_MAPPINGS = {
    "SeedVR2ModelLoader": "SeedVR2 Model Loader",
    "SeedVR2VAELoader": "SeedVR2 VAE Loader"
}

# Optional: Export mappings for transparency
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']