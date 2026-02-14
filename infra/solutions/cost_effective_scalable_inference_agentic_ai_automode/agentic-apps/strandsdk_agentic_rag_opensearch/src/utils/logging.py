"""Logging utilities for the application."""

import logging
import sys
from typing import Optional

def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )

def log_title(title: str, width: int = 60) -> None:
    """Print a formatted title for logging."""
    border = "=" * width
    padding = (width - len(title) - 2) // 2
    formatted_title = f"{border}\n{' ' * padding} {title} {' ' * padding}\n{border}"
    print(formatted_title)
