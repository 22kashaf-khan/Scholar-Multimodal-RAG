"""Structured logging setup using structlog.

Call configure_logging() or setup_logging() once at application startup.
All modules obtain a logger via structlog.get_logger(__name__).
"""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(level: str = "INFO", json_logs: bool = True) -> None:
    """Configure structlog with JSON output for production, pretty for dev.

    Args:
        level: Minimum log level (INFO, DEBUG, WARNING, ERROR).
        json_logs: True → machine-readable JSON (production/CI).
                   False → human-readable console output (CLI/dev).
    """
    processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    if json_logs:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=True,
    )


def setup_logging(json_logs: bool = True, level: str = "INFO") -> None:
    """Alias for configure_logging; preferred name for script/CLI entry points.

    Args:
        json_logs: False for human-readable console output (CLI mode).
        level: Minimum log level.
    """
    configure_logging(level=level, json_logs=json_logs)
