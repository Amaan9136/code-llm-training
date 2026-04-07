from __future__ import annotations
import logging
import sys
from pathlib import Path
from rich.logging import RichHandler
from rich.console import Console
console = Console()
def get_logger(name: str, level: str = "INFO", log_file: str = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    rich_handler = RichHandler(console=console, rich_tracebacks=True, markup=True)
    rich_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.addHandler(rich_handler)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        ))
        logger.addHandler(file_handler)
    return logger