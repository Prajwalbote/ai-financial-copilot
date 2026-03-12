"""
utils/config.py — Configuration Management
"""

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def load_config(config_path: str = "config.yaml") -> dict:
    """Load the YAML configuration file."""

    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(
            f"Config file not found at: {config_file.absolute()}\n"
            f"Make sure you're running from the project root directory."
        )

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    return config


class Config:
    """
    Centralized configuration object.

    Usage:
        from utils.config import Config
        cfg = Config()
        print(cfg.embedding.model_name)
    """

    def __init__(self, config_path: str = "config.yaml"):
        self._config = load_config(config_path)
        for key, value in self._config.items():
            if isinstance(value, dict):
                setattr(self, key, self._dict_to_obj(value))
            else:
                setattr(self, key, value)

    def _dict_to_obj(self, d: dict):
        """Convert a dictionary to object for dot-notation access."""

        class ConfigSection:
            pass

        obj = ConfigSection()
        for key, value in d.items():
            if isinstance(value, dict):
                setattr(obj, key, self._dict_to_obj(value))
            else:
                setattr(obj, key, value)
        return obj

    def get(self, key_path: str, default=None):
        """
        Access config via dot-notation string.
        Example: cfg.get('embedding.model_name')
        """

        keys = key_path.split('.')
        value = self._config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def get_env(self, key: str, default: str = None) -> str:
        """
        Get a secret from environment variables.
        Secrets NEVER go in config.yaml — they live in .env
        """

        value = os.environ.get(key, default)
        if value is None:
            raise EnvironmentError(
                f"Required environment variable '{key}' is not set.\n"
                f"Add it to your .env file: {key}=your_value_here"
            )
        return value

    def validate(self):
        """
        Check all required config sections exist.
        Always validate config at startup — catch problems
        early before they cause mysterious failures later.
        """

        required = ["embedding", "llm", "chunking", "vector_db"]
        for key in required:
            if not hasattr(self, key):
                raise ValueError(
                    f"Missing required config section: '{key}'\n"
                    f"Add it to your config.yaml file."
                )
        print("✅ Config validation passed!")


# Singleton — read once, shared everywhere
_config_instance = None


def get_config(config_path: str = "config.yaml") -> Config:
    """Get the singleton Config instance."""

    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_path)
    return _config_instance