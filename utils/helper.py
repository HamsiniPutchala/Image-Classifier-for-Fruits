"""Helper utilities for the project."""

import os
import logging
from typing import Optional
import json
from datetime import datetime

logger = logging.getLogger(__name__)


def create_directories(paths: list) -> None:
    """
    Create multiple directories if they don't exist.
    
    Args:
        paths (list): List of directory paths to create
    """
    for path in paths:
        os.makedirs(path, exist_ok=True)
        logger.info(f"Directory ensured: {path}")


def save_config(config: dict, save_path: str) -> None:
    """
    Save configuration to JSON file.
    
    Args:
        config (dict): Configuration dictionary
        save_path (str): Path to save JSON file
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=4)
    logger.info(f"Configuration saved to {save_path}")


def load_config(config_path: str) -> Optional[dict]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path (str): Path to JSON configuration file
        
    Returns:
        Optional[dict]: Configuration dictionary or None if file doesn't exist
    """
    if not os.path.exists(config_path):
        logger.warning(f"Configuration file not found: {config_path}")
        return None
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    logger.info(f"Configuration loaded from {config_path}")
    return config


def get_timestamp() -> str:
    """
    Get current timestamp in standard format.
    
    Returns:
        str: Formatted timestamp
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_message(message: str, level: str = "info") -> None:
    """
    Log a message with timestamp.
    
    Args:
        message (str): Message to log
        level (str): Log level ('info', 'warning', 'error')
    """
    timestamp = get_timestamp()
    log_msg = f"[{timestamp}] {message}"
    
    if level.lower() == "info":
        logger.info(log_msg)
    elif level.lower() == "warning":
        logger.warning(log_msg)
    elif level.lower() == "error":
        logger.error(log_msg)