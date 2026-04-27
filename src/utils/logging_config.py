"""
Logging configuration for the HR Taxonomy Pipeline.

Usage:
    from src.utils.logging_config import get_logger

    logger = get_logger(__name__)
    logger.info("Pipeline started")
    logger.error("Something went wrong", exc_info=True)
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

# ── Constants ────────────────────────────────────────────────────────────────

LOG_DIR = Path("logs")
LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
MAX_BYTES = 10 * 1024 * 1024  # 10 MB per log file
BACKUP_COUNT = 5               # Keep last 5 rotated files

# Shared Rich console (used for pretty terminal output anywhere in the app)
console = Console()

# ── Setup ─────────────────────────────────────────────────────────────────────


def setup_logging(
    log_level: str = "INFO",
    log_dir: Path = LOG_DIR,
) -> None:
    """
    Configure root logger with:
      - Rich handler  → pretty, color-coded console output
      - RotatingFileHandler → persistent logs/pipeline.log
      - FileHandler (costs) → logs/costs.log for token/cost tracking

    Call once at application entry point (e.g., top of main scripts).

    Args:
        log_level: One of DEBUG, INFO, WARNING, ERROR, CRITICAL.
        log_dir:   Directory where log files are written.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, log_level.upper(), logging.INFO)

    # ── Root logger ───────────────────────────────────────────────────────────
    root = logging.getLogger()
    root.setLevel(level)

    # Remove any handlers already attached (avoids duplicate logs in notebooks)
    root.handlers.clear()

    # ── Handler 1: Rich console ───────────────────────────────────────────────
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=True,
        markup=True,
        rich_tracebacks=True,
        tracebacks_show_locals=True,
    )
    rich_handler.setLevel(level)
    root.addHandler(rich_handler)

    # ── Handler 2: Rotating file (main log) ──────────────────────────────────
    main_log_path = log_dir / "pipeline.log"
    file_handler = logging.handlers.RotatingFileHandler(
        filename=main_log_path,
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    root.addHandler(file_handler)

    # ── Handler 3: Cost tracking log ─────────────────────────────────────────
    cost_logger = logging.getLogger("costs")
    cost_log_path = log_dir / "costs.log"
    cost_handler = logging.FileHandler(cost_log_path, encoding="utf-8")
    cost_handler.setLevel(logging.INFO)
    cost_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(message)s", datefmt=DATE_FORMAT)
    )
    cost_logger.addHandler(cost_handler)
    cost_logger.propagate = False  # Don't duplicate to root

    logging.getLogger(__name__).info(
        f"Logging initialised | level={log_level} | file={main_log_path}"
    )


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger.  Always call setup_logging() once before using loggers.

    Args:
        name: Typically __name__ of the calling module.

    Returns:
        logging.Logger instance.
    """
    return logging.getLogger(name)


def get_cost_logger() -> logging.Logger:
    """Return the dedicated cost-tracking logger."""
    return logging.getLogger("costs")