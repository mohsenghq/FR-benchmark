# common/utils.py
import os
import logging

logger = logging.getLogger(__name__)

def ensure_dir_exists(dir_path: str):
    """Creates a directory if it doesn't already exist."""
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
        except OSError as e:
            logger.error(f"Error creating directory {dir_path}: {e}")
            raise


