import os
import yaml
from typing import Dict, Any

class ConfigLoader:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found at: {self.config_path}")
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    def get(self, key_path: str, default: Any = None) -> Any:
        keys = key_path.split(".")
        val = self._config
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

def get_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    return ConfigLoader(config_path).config
