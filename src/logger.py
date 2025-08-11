import logging
import os
from datetime import datetime
from config.constants import LOG_FILE_PATH

def setup_logging():
    """
    Sets up the logging configuration for the MLOps project.
    Logs messages to both console and a file.
    """
    # Ensure the log directory exists
    log_dir = os.path.dirname(LOG_FILE_PATH)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) # Set the minimum level for logging

    # Check if handlers are already added to prevent duplicate logs
    if not logger.handlers:
        # Create a file handler
        file_handler = logging.FileHandler(LOG_FILE_PATH)
        file_handler.setLevel(logging.INFO)

        # Create a console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create a formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add the handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

# Initialize logger for use across modules
logger = setup_logging()