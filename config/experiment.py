import json
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).with_name("experiment_config.json")


def get_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


CONFIG = get_config()


def get_nested(*keys: str, default: Any = None) -> Any:
    value: Any = CONFIG
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value
