from pathlib import Path

import torch


class Settings:
    PACKAGE_ROOT_DIR = Path(__file__).parent
    CUSTOM_MODELS_DIR = PACKAGE_ROOT_DIR.parent / "custom_models"
    TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


settings = Settings()
