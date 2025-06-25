"""
logging.py

Simple logging utility for the MLOps project.
"""

import logging
import sys
from typing import Optional


def setup_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        level (str): Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        format_string (Optional[str]): Custom format string for log messages
        log_file (Optional[str]): Path to log file (if None, logs to console)

    Returns:
        logging.Logger: Configured logger instance
    """
    if format_string is None:
        format_string = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"

    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format=format_string,
        handlers=[
            logging.StreamHandler(sys.stdout),
            *([logging.FileHandler(log_file)] if log_file else []),
        ],
    )

    return logging.getLogger(__name__)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Args:
        name (str): Name for the logger (typically __name__)

    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)
