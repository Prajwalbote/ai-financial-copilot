"""
utils/logger.py — Centralized Logging
======================================
WHY LOGGING MATTERS:
  print() statements disappear. Logs persist to files,
  have timestamps, severity levels, and can be shipped
  to monitoring tools in production.

  A senior engineer NEVER uses print() in production.
  They use loggers.

LOG LEVELS GUIDE:
  DEBUG   — Detailed internals (development only)
  INFO    — Normal operations ("Loaded 50 documents")
  WARNING — Unexpected but not breaking ("Chunk too small")
  ERROR   — Something failed ("Failed to parse PDF")
  CRITICAL— System cannot continue ("DB connection lost")
"""

import sys
from pathlib import Path
from loguru import logger


def setup_logger(
    log_level: str = "INFO",
    log_file: str = "logs/app.log",
    rotation: str = "10 MB",
    retention: str = "7 days",
):
    """
    Configure the application logger.
    
    Args:
        log_level: Minimum level to log
        log_file:  Path to log file
        rotation:  When to create a new log file
        retention: How long to keep old logs
    """
    
    # Remove default handler to avoid duplicate logs
    logger.remove()
    
    # Console handler — colorized, readable in development
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan> | "
               "<level>{message}</level>",
        level=log_level,
        colorize=True,
    )
    
    # File handler — persistent logs for production
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} | {message}",
        level=log_level,
        rotation=rotation,
        retention=retention,
        compression="zip",
    )


def get_logger(module_name: str):
    """
    Get a logger for a specific module.
    
    Usage:
        from utils.logger import get_logger
        log = get_logger(__name__)
        log.info("Processing document...")
        log.error(f"Failed to parse file: {str(e)}")
    
    WHY MODULE NAME?
    Logs will show exactly which file the message came from.
    Critical when debugging production issues.
    """
    return logger.bind(name=module_name)


# Initialize logger on import
try:
    from utils.config import get_config
    cfg = get_config()
    setup_logger(
        log_level=cfg.logging.level,
        log_file=cfg.logging.log_file
    )
except Exception:
    # Config may not be ready yet on first setup
    setup_logger()