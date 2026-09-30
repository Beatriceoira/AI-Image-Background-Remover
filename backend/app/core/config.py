import os

# Class-level configuration - singleton pattern
class Config:
    # Model configuration
    MODEL_NAME: str = os.getenv("MODEL_NAME", "u2net")
    MODEL_PATH: str = os.getenv(
        "MODEL_PATH",
        os.path.join(os.getcwd(), "backend", "app", "models", "weights", f"{MODEL_NAME}.pth")
    )
    # Device selection
    ENABLE_GPU: bool = os.getenv("ENABLE_GPU", "True").lower() in ("true", "1", "yes")
    # Upload limits
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "15"))
    MAX_IMAGE_WIDTH: int = int(os.getenv("MAX_IMAGE_WIDTH", "6000"))
    MAX_IMAGE_HEIGHT: int = int(os.getenv("MAX_IMAGE_HEIGHT", "6000"))
    # Output format
    OUTPUT_FORMAT: str = os.getenv("OUTPUT_FORMAT", "PNG").upper()
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

    @classmethod
    def get_device(cls) -> str:
        """Return 'cuda' if GPU enabled and available, else 'cpu'."""
        if cls.ENABLE_GPU:
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
            except Exception:
                pass
        return "cpu"