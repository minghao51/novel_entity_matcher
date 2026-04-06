"""Centralized logging configuration for novel_entity_matcher.

This module provides a unified logging infrastructure with proper Python logging
module usage, third-party library log suppression, and environment variable support.
"""

import logging
import os
import warnings
from typing import Optional

# Global state to track if logging has been configured
_logging_configured = False


def configure_logging(
    verbose: bool = False,
    log_level: Optional[int] = None,
    log_file: Optional[str] = None,
) -> None:
    """Configure logging for novel_entity_matcher package.

    This function sets up the root logger for the novel_entity_matcher package
    with appropriate handlers, formatters, and log levels. It can be called
    multiple times - subsequent calls will update the log level.

    Args:
        verbose: If True, set log level to DEBUG with detailed formatting.
                 If False, set log level to WARNING with simple formatting.
        log_level: Optional explicit log level (overrides verbose parameter).
                   Can be DEBUG, INFO, WARNING, or ERROR.
        log_file: Optional file path to write logs in addition to console.

    Example:
        >>> from novelentitymatcher.utils.logging_config import configure_logging
        >>> configure_logging(verbose=True)  # Enable detailed logging
        >>> configure_logging(verbose=False)  # Quiet mode (default)
    """
    global _logging_configured

    # Determine log level
    if log_level is None:
        log_level = logging.DEBUG if verbose else logging.WARNING

    # Get the novel_entity_matcher root logger
    logger = logging.getLogger("novelentitymatcher")

    # Format string - verbose mode includes level and module name
    if verbose:
        format_str = "[%(levelname)s] %(name)s: %(message)s"
    else:
        # Simple format for quiet mode
        format_str = "%(message)s"

    formatter = logging.Formatter(format_str)

    # If this is the first configuration, set up handlers
    if not _logging_configured:
        logger.setLevel(log_level)

        # Remove any existing handlers to avoid duplicates
        logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)

        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Optional file handler for debugging
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        # Suppress third-party loggers in quiet mode
        if not verbose:
            suppress_third_party_loggers()

        _logging_configured = True
    else:
        # Already configured, update levels and formatter to match the latest mode.
        logger.setLevel(log_level)
        for handler in logger.handlers:
            handler.setLevel(log_level)
            handler.setFormatter(formatter)

        if log_file and not any(
            isinstance(handler, logging.FileHandler)
            and getattr(handler, "baseFilename", None) == os.path.abspath(log_file)
            for handler in logger.handlers
        ):
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        if not verbose:
            suppress_third_party_loggers()


def suppress_third_party_loggers() -> None:
    """Suppress verbose logging from third-party ML libraries.

    This function sets the log level of common third-party libraries to WARNING
    to reduce noise during normal operation. It also filters specific warnings
    from PyTorch and transformers.

    The suppressed libraries include:
    - sentence_transformers: Model loading and inference logs
    - transformers: Training and model loading logs
    - setfit: Training progress logs
    - torch: CUDA/device information
    - datasets: Data processing logs
    - huggingface_hub: Download/cache logs
    - urllib3: HTTP connection logs
    - PIL: Image processing logs

    Example:
        >>> from novelentitymatcher.utils.logging_config import suppress_third_party_loggers
        >>> suppress_third_party_loggers()
    """
    # List of loggers to suppress
    loggers_to_suppress = [
        "sentence_transformers",
        "transformers",
        "setfit",
        "torch",
        "datasets",
        "huggingface_hub",
        "urllib3",
        "PIL",
        "PIL.PngImagePlugin",
        "PIL.Image",
    ]

    for logger_name in loggers_to_suppress:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.WARNING)

    # Suppress specific warnings from third-party libraries
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
    warnings.filterwarnings(
        "ignore", category=FutureWarning, module="sentence_transformers"
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module.

    This function returns a logger instance with the proper namespace
    for the novel_entity_matcher package. Use this instead of logging.getLogger()
    to ensure consistent logger naming.

    Args:
        name: Module name (typically __name__).

    Returns:
        A configured logger instance.

    Example:
        >>> from novelentitymatcher.utils.logging_config import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Starting operation")
        >>> logger.debug("Detailed information")
    """
    if name == "novelentitymatcher" or name.startswith("novelentitymatcher."):
        return logging.getLogger(name)
    return logging.getLogger(f"novelentitymatcher.{name}")


def set_log_level(level: int) -> None:
    """Change log level at runtime.

    This function allows dynamic changes to the log level after initial
    configuration. Useful for temporarily enabling debug logging.

    Args:
        level: New log level (DEBUG, INFO, WARNING, or ERROR).

    Example:
        >>> from novelentitymatcher.utils.logging_config import set_log_level, DEBUG
        >>> set_log_level(DEBUG)  # Enable debug logging
        >>> logger.debug("This will now be shown")
    """
    logger = logging.getLogger("novelentitymatcher")

    # Update logger level
    logger.setLevel(level)

    # Update all handlers
    for handler in logger.handlers:
        handler.setLevel(level)
