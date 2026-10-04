"""Load config.yaml (every pre-registered number lives there)."""
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).with_name("config.yaml")


def load_config(path=CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
