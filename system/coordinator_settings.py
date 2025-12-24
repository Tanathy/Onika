from typing import Any, Dict
import json
import re
from pathlib import Path

# Global settings dictionary, populated by run.py or config manager
SETTINGS: Dict[str, Any] = {}
PACKAGES: Dict[str, Any] = {}

def reload_packages():
    """
    Reloads packages from config/packages.jsonc into PACKAGES global.
    Handles comments in JSONC.
    """
    global PACKAGES
    
    # Assuming ROOT is 2 levels up from this file (system/coordinator_settings.py -> system/ -> root)
    # But run.py sets sys.path, so we can find config relative to CWD or __file__
    # run.py runs from ROOT usually.
    
    root_path = Path(__file__).resolve().parents[1]
    packages_path = root_path / "config" / "packages.jsonc"
    
    if not packages_path.exists():
        print(f"[WARNING] packages.jsonc not found at {packages_path}")
        PACKAGES = {}
        return

    try:
        with open(packages_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Remove comments
        # Simple regex for // comments and /* */ comments
        # Note: This is a simple parser, might not handle all edge cases (strings with //)
        content = re.sub(r"//.*", "", content)
        content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)
        
        PACKAGES = json.loads(content)
        # print(f"[INFO] Loaded packages configuration.")
    except Exception as e:
        print(f"[ERROR] Failed to load packages.jsonc: {e}")

def reload_settings() -> Dict[str, Any]:
    """
    Reloads settings from config/configs.json into SETTINGS global.
    Returns the SETTINGS dict.
    """
    global SETTINGS
    
    root_path = Path(__file__).resolve().parents[1]
    config_path = root_path / "config" / "configs.json"
    
    if not config_path.exists():
        print(f"[WARNING] configs.json not found at {config_path}")
        return SETTINGS

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            SETTINGS.update(json.load(f))
    except Exception as e:
        print(f"[ERROR] Failed to load configs.json: {e}")
        
    return SETTINGS

