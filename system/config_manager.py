import json
import os
from pathlib import Path
from typing import Any, Dict
from system import coordinator_settings as cs
from system.log import info, warning, error

def deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries.
    """
    for key, value in update.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base

def load_config(root_path: Path) -> None:
    """
    Load configs.json and project/settings.json, merge them, and update cs.SETTINGS.
    """
    config_path = root_path / "config" / "configs.json"
    project_settings_path = root_path / "project" / "settings.json"

    # Load default config
    if not config_path.exists():
        error(f"Default config not found at {config_path}")
        default_config = {}
    else:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                default_config = json.load(f)
            info(f"Loaded default config from {config_path}")
        except Exception as e:
            error(f"Failed to load default config: {e}")
            default_config = {}

    # Load project settings
    project_config = {}
    if project_settings_path.exists():
        try:
            with open(project_settings_path, "r", encoding="utf-8") as f:
                project_config = json.load(f)
            info(f"Loaded project settings from {project_settings_path}")
        except Exception as e:
            warning(f"Failed to load project settings: {e}")

    # Merge
    merged_config = deep_merge(default_config, project_config)
    
    # Update global settings
    cs.SETTINGS.update(merged_config)
    info("Configuration loaded and merged.")
