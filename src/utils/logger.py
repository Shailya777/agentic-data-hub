import os
import logging
import sys


def setup_logger(name: str= 'agentic_hub') -> logging.Logger:
    """
    Sets up a dual-destination logger that outputs clean logs
    to both the terminal console and a tracking log file.
    :param name: Name of the logger.
    :return: Logger object
    """

    logger= logging.getLogger(name)

    # Preventing Duplicate Handlers is Logger is called multiple times across modules:
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)

    # Creating Log Directory if it doesn't exist:
    log_dir= 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # File Handler (Detailed Metadata)
    file_handler= logging.FileHandler(os.path.join(log_dir, 'hub_execution.log'), encoding='utf-8')
    file_formatter= logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d]: %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    file_handler.setFormatter(file_formatter)

    # Console Handler (Clean, Scannable Terminal Tracking):
    console_handler= logging.StreamHandler()
    console_formatter= logging.Formatter("[%(levelname)s] %(message)s")
    console_handler.setFormatter(console_formatter)

    # Registering Handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# Logger Instance importable across Entire Project:
hub_logger= setup_logger()