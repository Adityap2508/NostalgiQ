"""
Configuration for media intelligence layer.
Use GPU if available; set USE_GPU=False to force CPU.
"""

import os

# Default: use CPU unless explicitly enabled and CUDA available
_use_gpu_env = os.environ.get("USE_GPU", "false").lower() in ("true", "1", "yes")


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# Only use GPU if explicitly requested AND CUDA is available
USE_GPU = _use_gpu_env and _cuda_available()
