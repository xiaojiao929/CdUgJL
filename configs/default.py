import yaml
from pathlib import Path


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def get_cfg_defaults():
    default_path = Path(__file__).parent / 'default.yaml'
    return load_config(str(default_path))
