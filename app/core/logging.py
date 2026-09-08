import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Настраивает корневой логгер на stdout (стандартный вывод)"""
    logging.basicConfig(
        level=level,
        stream=sys.stdout,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )