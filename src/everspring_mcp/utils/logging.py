"""EverSpring MCP - Centralized logging configuration.

All application logs are stored in the logs/ directory with:
- Daily rotation
- Readable format with timestamps
- Console + file output
- Separate files for different components
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Project root and logs directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure logs directory exists
LOGS_DIR.mkdir(exist_ok=True)

# Log format - readable with timestamp, level, logger name, and message
LOG_FORMAT = "%(asctime)s [%(levelname)-8s] %(name)-30s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# File settings
MAX_BYTES = 10 * 1024 * 1024  # 10 MB per file
BACKUP_COUNT = 5  # Keep 5 backup files


def get_log_file(name: str = "everspring") -> Path:
    """Get log file path with today's date.
    
    Args:
        name: Base name for log file
        
    Returns:
        Path to log file (e.g., logs/everspring_2024-01-15.log)
    """
    today = datetime.now().strftime("%Y-%m-%d")
    return LOGS_DIR / f"{name}_{today}.log"


def setup_logging(
    level: str | int = logging.INFO,
    console: bool = True,
    file: bool = True,
    name: str = "everspring",
) -> logging.Logger:
    """Configure centralized logging for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        console: Whether to log to console (stderr)
        file: Whether to log to file
        name: Base name for log file
        
    Returns:
        Root logger configured with handlers
    """
    # Get the root logger for everspring_mcp
    root_logger = logging.getLogger("everspring_mcp")
    
    # Convert string level to int
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    
    # Console handler (stderr to avoid interfering with MCP stdio)
    if console:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler with rotation
    if file:
        log_file = get_log_file(name)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=MAX_BYTES,
            backupCount=BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        # Log startup message
        root_logger.info(f"Logging to: {log_file}")
    
    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module.
    
    Args:
        name: Module name (will be prefixed with everspring_mcp)
        
    Returns:
        Logger instance
    """
    if not name.startswith("everspring_mcp"):
        name = f"everspring_mcp.{name}"
    return logging.getLogger(name)


# Quick setup for imports
def configure(level: str = "INFO") -> None:
    """Quick configuration helper.
    
    Usage:
        from everspring_mcp.utils.logging import configure
        configure("DEBUG")
    """
    setup_logging(level=level)


__all__ = [
    "setup_logging",
    "get_logger",
    "configure",
    "LOGS_DIR",
    "get_log_file",
]
