#!/usr/bin/env python3
"""
Tools for loading and merging configuration files.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional


def load_base_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    flattened = flatten_config(config)
    
    return flattened


def flatten_config(config: Dict, parent_key: str = '') -> Dict:
    """
    Args:
        config; parent_key.
    
    Returns:
        a flattened dictionary with dot-separated keys
    """
    items = []
    for k, v in config.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_config(v, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)


def save_config(config: Dict, save_path: Path):
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2)


def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    merged = base_config.copy()
    merged.update(override_config)
    return merged