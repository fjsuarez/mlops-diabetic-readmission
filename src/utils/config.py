import os
import yaml


def load_config(
    logger,
    config_path: str = "config.yaml",
) -> dict:
    """
    Loads configuration from the given YAML file.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    logger.info(f"Loading config from: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config
