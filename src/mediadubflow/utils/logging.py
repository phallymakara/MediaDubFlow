"""
Structured logging configuration using loguru.

Call setup_logging() once at application startup.  All modules should
use ``from loguru import logger`` directly — no need to pass logger
instances around.
"""

from __future__ import annotations

import sys
from pathlib import Path


def setup_logging(
    log_level: str = "INFO",
    log_dir: Path = Path("./logs"),
    log_json: bool = False,
) -> None:
    """
    Configure loguru with:
    - A human-readable sink to stderr for development.
    - A rotating file sink (one per day) for persistent logs.
    - Optional JSON output for structured log ingestion.

    Args:
        log_level: Minimum log level (DEBUG / INFO / WARNING / ERROR).
        log_dir:   Directory where log files are written.
        log_json:  If True, emit JSON-formatted log records.
    """
    from loguru import logger  # noqa: PLC0415

    # Remove the default loguru handler.
    logger.remove()

    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )

    # --- Stderr sink ---
    if log_json:
        logger.add(sys.stderr, level=log_level, serialize=True)
    else:
        logger.add(sys.stderr, level=log_level, format=log_format, colorize=True)

    # --- File sink ---
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "mediadubflow_{time:YYYY-MM-DD}.log"

    if log_json:
        logger.add(
            str(log_file),
            level=log_level,
            rotation="00:00",
            retention="30 days",
            compression="gz",
            serialize=True,
            enqueue=True,  # thread-safe non-blocking writes
        )
    else:
        logger.add(
            str(log_file),
            level=log_level,
            format=log_format,
            rotation="00:00",
            retention="30 days",
            compression="gz",
            enqueue=True,
        )

    logger.info("Logging configured — level={}, json={}", log_level, log_json)
