import sys

from loguru import logger

FORMAT = "<level>{level.name}</level> | " "<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"


def set_logger_style():
    logger.remove()
    logger.add(sys.stdout, format=FORMAT, level="TRACE", filter=lambda record: "compile_log" not in record["extra"])
