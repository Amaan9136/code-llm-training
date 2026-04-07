from __future__ import annotations
import os
from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field
import yaml
class Settings(BaseSettings):
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = Field(default_factory=lambda: Path("data"))
    OUTPUTS_DIR: Path = Field(default_factory=lambda: Path("outputs"))
    LOGS_DIR: Path = Field(default_factory=lambda: Path("logs"))
    CONFIG_PATH: Path = Field(default_factory=lambda: Path("config/training.yaml"))
    DATABASE_URL: str = "sqlite:///data/codellm.db"
    REDIS_URL: str = "redis://localhost:6379/0"
    HF_TOKEN: Optional[str] = None
    WANDB_API_KEY: Optional[str] = None
    MAX_WORKERS: int = 4
    DEBUG: bool = False
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
    def model_post_init(self, __context):
        for d in [self.DATA_DIR, self.OUTPUTS_DIR, self.LOGS_DIR,
                  self.DATA_DIR / "raw", self.DATA_DIR / "processed",
                  self.DATA_DIR / "dataset"]:
            Path(d).mkdir(parents=True, exist_ok=True)
def load_config(path: Optional[str] = None) -> dict:
    config_path = path or os.environ.get("CONFIG_PATH", "config/training.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
settings = Settings()