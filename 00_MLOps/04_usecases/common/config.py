# common/config.py
"""Configuration loader for MLOps pipelines."""

import yaml
from pathlib import Path

_config = None


def get_config() -> dict:
    """Load configuration from config.yaml."""
    global _config
    if _config is None:
        config_path = Path(__file__).parent.parent / "config.yaml"
        with open(config_path) as f:
            _config = yaml.safe_load(f)
    return _config
