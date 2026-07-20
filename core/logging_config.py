import logging
import sys
from logging.handlers import RotatingFileHandler
from core.config import LOG_FILE


def setup_logger() -> logging.Logger:
    """Configures a root logger for both file and terminal output."""
    logger = logging.getLogger("localRAGvault")

    # Avoid duplicate handlers if initialized multiple times
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(name)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Rotating File Handler (Max 5MB per file, keeping 3 backups)
    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Terminal Stream Handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


# Globally accessible logger instance
logger = setup_logger()
