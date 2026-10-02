"""Configuration loader helper."""

import yaml


def get_config(config_path: str = "config.yaml") -> dict:
    """Read and return master YAML configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
