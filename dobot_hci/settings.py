from pathlib import Path


class Settings:
    PACKAGE_ROOT_DIR = Path(__file__).parent
    CUSTOM_MODELS_DIR = PACKAGE_ROOT_DIR.parent / "custom_models"


settings = Settings()
