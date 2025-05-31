import json
import os
import yaml


def load_json(path: str) -> dict:
    """Load a JSON file from a given path."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: str) -> dict:
    """Load a YAML file from a given path."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(data: dict, path: str) -> None:
    """Save a dictionary to a JSON file."""
    # Ensure directory exists, handle potential empty path for dirname
    dir_name = os.path.dirname(path)
    if dir_name:  # Only try to create directory if path includes one
        os.makedirs(dir_name, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
