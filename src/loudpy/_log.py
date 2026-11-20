"""
Messages of loudpy: the standard `logging` module, logger "loudpy" (and children "loudpy.mesh",
"loudpy.study", …). Progress messages are INFO, solver details DEBUG.

    lp.set_verbosity("info")      default: progress of meshing, sweeps, mode searches
    lp.set_verbosity("debug")     solver details (shifts, filters, mode tables)
    lp.set_verbosity("warning")   quiet
"""
from __future__ import annotations

import logging

logger = logging.getLogger("loudpy")
_handler: logging.Handler | None = None


def set_verbosity(level="info") -> None:
    """Level of the messages printed by loudpy: "debug", "info", "warning", "error" (or a logging level)."""
    global _handler
    if _handler is None:
        _handler = logging.StreamHandler()
        _handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(_handler)
        logger.propagate = False
    logger.setLevel(level.upper() if isinstance(level, str) else level)


def info(msg, *args, **kwargs):
    """A progress callback (the log= argument of the sweeps): logs msg at INFO."""
    logger.info(str(msg) % args if args else str(msg))
