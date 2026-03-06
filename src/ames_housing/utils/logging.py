"""Structured logging via Loguru — single setup point for the whole app."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from ames_housing.config import settings


def setup_logging(level: str | None = None) -> None:
    """Configure Loguru for console + rotating file output."""
    cfg = settings.logging
    effective_level = level or cfg.level

    # ── Remove default handler ────────────────────────────────────────────────
    logger.remove()

    # ── Console (stdout) ──────────────────────────────────────────────────────
    logger.add(
        sys.stdout,
        level=effective_level,
        format=cfg.format,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # ── Rotating file ─────────────────────────────────────────────────────────
    log_path = Path(cfg.file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        str(log_path),
        level=effective_level,
        format=cfg.format,
        rotation=cfg.rotation,
        retention=cfg.retention,
        compression="zip",
        backtrace=True,
        diagnose=True,
        enqueue=True,          # thread-safe async write
    )

    logger.info(
        "Logging configured | level={} | file={}",
        effective_level,
        log_path,
    )


def get_logger(name: str = __name__):
    """Return a contextualised logger bound with the caller's module name."""
    return logger.bind(module=name)
