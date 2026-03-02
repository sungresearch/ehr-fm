import logging
import sys


def setup_logging(level: str = "INFO", child_name: str = None) -> logging.Logger:
    """Return (or create) a logger under the ``ehr-fm`` namespace."""
    logger_name = f"ehr-fm.{child_name}" if child_name else "ehr-fm"
    logger = logging.getLogger(logger_name)
    logger.setLevel(level.upper())

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level.upper())
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.propagate = False
    else:
        new_level = getattr(logging, level.upper(), logging.INFO)
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(min(handler.level, new_level))
        logger.setLevel(min(logger.level, new_level))

    return logger
