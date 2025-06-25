"""
data_loader.py

Standalone data loading module that reads configuration from config.yaml
and returns a pandas DataFrame.
"""

import os
import pandas as pd
import sys
from pathlib import Path
from src.utils.config import load_config
from src.utils.logging import setup_logging, get_logger


class DataLoader:
    """
    Data loader class that handles loading data based on configuration.
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the DataLoader with configuration.

        Args:
            config_path (str): Path to the configuration file
        """
        self.logger = get_logger(__name__)

        # Load configuration
        try:
            self.config = load_config(self.logger, config_path)
            self.data_config = self.config.get("data_source", {})
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise

    def load_data(self) -> pd.DataFrame:
        """Load data based on the configuration settings."""
        file_path = self.data_config.get("raw_path")
        file_type = self.data_config.get("type", "csv").lower()

        if not file_path:
            raise ValueError("No file path specified in configuration")

        # Resolve path relative to project root if it's relative
        if not os.path.isabs(file_path):
            # Get project root (assuming data_loader.py is in src/data_load/)
            project_root = Path(__file__).resolve().parents[2]
            file_path = str(project_root / file_path)

        # Check if file exists
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")

        self.logger.info(f"Loading data from: {file_path}")

        try:
            if file_type == "csv":
                df = self._load_csv(file_path)
            elif file_type == "excel" or file_type == "xlsx":
                df = self._load_excel(file_path)
            elif file_type == "json":
                df = self._load_json(file_path)
            elif file_type == "parquet":
                df = self._load_parquet(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")

            self.logger.info(
                    f"Successfully loaded data with shape: {df.shape}"
                )
            return df

        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            raise

    def _load_csv(self, file_path: str) -> pd.DataFrame:
        """Load CSV file with configuration parameters."""
        return pd.read_csv(
            file_path,
            delimiter=self.data_config.get("delimiter", ","),
            header=self.data_config.get("header", 0),
            encoding=self.data_config.get("encoding", "utf-8"),
        )

    def _load_excel(self, file_path: str) -> pd.DataFrame:
        """Load Excel file."""
        return pd.read_excel(
                file_path, header=self.data_config.get("header", 0)
            )


def load_diabetic_data(config_path: str = "config.yaml") -> pd.DataFrame:
    """
    Convenience function to load diabetic readmission data.

    Args:
        config_path (str): Path to configuration file

    Returns:
        pd.DataFrame: Loaded data
    """
    setup_logging()

    loader = DataLoader(config_path)
    return loader.load_data()


if __name__ == "__main__":
    logger = setup_logging()

    try:
        df = load_diabetic_data()
        print(f"Data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)
