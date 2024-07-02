import os
from pathlib import Path

import torch

TORCH_DEVICE = "cpu"

if torch.cuda.is_available():
    TORCH_DEVICE = "cuda"
elif torch.backends.mps.is_available():
    TORCH_DEVICE = "mps"
