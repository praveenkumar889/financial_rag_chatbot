import logging
import sys
import os
from logging.handlers import RotatingFileHandler

def setup_logger(level=logging.INFO, log_file="financial_agent.log"):
    """
    Configures the ROOT logger for production visibility.
    
    Features:
    - Root Logger: Captures logs from all modules.
    - Console Handler: CLEAN standard output (INFO+).
    - File Handler: DETAILED rotating logs (DEBUG+).
    - Prevents duplicate handlers.
    """
    logger = logging.getLogger()
    logger.setLevel(level)

    if logger.handlers:
        logger.handlers.clear()

    # --- Console Handler (Standard/No Color) ---
    console_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        datefmt="%H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # --- File Handler (Rotating, DEBUG+) ---
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_path = os.path.join(base_dir, log_file)

    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = RotatingFileHandler(
        log_path, 
        maxBytes=5*1024*1024, 
        backupCount=3, 
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG) 
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger
