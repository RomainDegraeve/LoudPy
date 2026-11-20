"""
Messages of loudpy: the standard `logging` module, logger "loudpy" (and children "loudpy.mesh",
"loudpy.study", …). Progress messages are INFO, solver details DEBUG.

loudpy is silent by default (a library does not configure logging). To see its messages:

    lp.set_verbosity("info")      progress of meshing, sweeps, mode searches
    lp.set_verbosity("debug")     solver details (shifts, filters, mode tables)
    lp.set_verbosity("warning")   only warnings (default of Python's logging)

or configure the "loudpy" logger like any other (logging.basicConfig, handlers, files …).
"""
from __future__ import annotations

import logging

logger = logging.getLogger("loudpy")
logger.addHandler(logging.NullHandler())          # silent unless the application configures logging
_handler: logging.Handler | None = None


def set_verbosity(level="info") -> None:
    """Print loudpy's messages from this level on: "debug", "info", "warning", "error" (or a logging level).
    Adds one console handler to the "loudpy" logger (once)."""
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
