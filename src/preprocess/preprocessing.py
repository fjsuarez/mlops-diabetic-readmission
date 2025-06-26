"""
preprocessing.py

Core data preprocessing pipeline - handles cleaning, missing values,
and basic transformations.
Integrates with the MLOps configuration system for consistency.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import List

# Add project root to Python path before importing from src
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Now import from src
from src.utils.config import load_config  # noqa: E402
from src.utils.logging import setup_logging, get_logger  # noqa: E402

logger = get_logger(__name__)

pd.set_option('future.no_silent_downcasting', True)


class DiabetesDataPreprocessor:
    """
    Core data preprocessing pipeline - handles cleaning, missing values, and
    basic transformations.
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the preprocessor with configuration.

        Args:
            config_path (str): Path to the configuration file
        """
        self.logger = get_logger(__name__)
        self.config_path = config_path

        # Load configuration
        try:
            self.config = load_config(self.logger, config_path)
            self.preprocess_config = self.config.get("preprocessing", {})
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise

        self.is_fitted = False

    def load_data(self, filepath: str = None) -> pd.DataFrame:
        """Load data with proper NA handling using configuration."""
        if filepath:
            # If specific filepath provided, load directly with config
            data_source_config = self.config.get("data_source", {})
            try:
                df = pd.read_csv(
                    filepath,
                    keep_default_na=data_source_config.get(
                        "keep_default_na", False),
                    na_values=data_source_config.get("na_values", [''])
                )
                self.logger.info(f"Data loaded successfully.\
                     Shape: {df.shape}")
                return df
            except Exception as e:
                self.logger.error(f"Error loading data: {e}")
                raise
        else:
            # Use DataLoader for consistency
            from src.data_load.data_loader import DataLoader  # noqa: E402
            loader = DataLoader(self.config_path)
            return loader.load_data()

    def clean_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values and clean data."""
        df = df.copy()

        # Handle empty DataFrame
        if df.empty:
            self.logger.info("Input DataFrame is empty, returning as-is")
            return df

        # Get configuration or use defaults
        drop_columns = self.preprocess_config.get('drop_columns', [])
        exclude_disposition = self.preprocess_config.get(
            'exclude_discharge_disposition', [])
        na_indicators = self.preprocess_config.get('na_indicators', [])

        # Convert '?' and 'Unknown/Invalid' to NaN
        if na_indicators:
            df.replace(na_indicators, np.nan, inplace=True)
            self.logger.info(f"Replaced NA indicators: {na_indicators}")

        # Drop columns with too many missing values or no information
        existing_drop_columns = [col for col in drop_columns
                                 if col in df.columns]
        if existing_drop_columns:
            df = df.drop(columns=existing_drop_columns)
            self.logger.info(f"Dropped columns: {existing_drop_columns}")

        # Drop rows with missing values
        initial_shape = df.shape
        df = df.dropna()
        self.logger.info(f"Dropped {initial_shape[0] - df.shape[0]} rows\
             with missing values")

        # Remove death cases (only if the column exists)
        if 'discharge_disposition_id' in df.columns:
            for disposition_id in exclude_disposition:
                initial_count = len(df)
                df = df[df['discharge_disposition_id'] != disposition_id]
                removed_count = initial_count - len(df)
                if removed_count > 0:
                    self.logger.info(f"Removed {removed_count} death cases\
                         (disposition_id={disposition_id})")

        self.logger.info(f"After cleaning: {df.shape}")
        return df

    def clean_diagnosis_codes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove decimal part from ICD-9 diagnosis codes."""
        df = df.copy()

        diag_cols = ['diag_1', 'diag_2', 'diag_3']
        for col in diag_cols:
            if col in df.columns:
                df[col] = df[col].apply(self._remove_after_period)

        self.logger.info("Cleaned diagnosis codes")
        return df

    def _remove_after_period(self, value):
        """Remove information after period in diagnosis codes."""
        if isinstance(value, str) and '.' in value:
            return value.split('.')[0]
        return value

    def remove_duplicates(
                          self, df: pd.DataFrame,
                          subset: List[str] = None) -> pd.DataFrame:
        """Remove duplicate patients, keeping first occurrence."""
        # Get subset from config or use default
        if subset is None:
            subset = self.preprocess_config.get(
                'duplicate_subset', ['patient_nbr'])

        initial_shape = df.shape
        df = df.drop_duplicates(subset=subset, keep='first')
        removed_count = initial_shape[0] - df.shape[0]
        if removed_count > 0:
            self.logger.info(f"Removed {removed_count} duplicate records\
                 based on {subset}")
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Complete preprocessing pipeline."""
        self.logger.info("Starting preprocessing pipeline...")

        # Check if preprocessing is enabled
        if not self.preprocess_config.get('enabled', True):
            self.logger.info("Preprocessing is disabled in config.")
            self.is_fitted = True  # Set fitted even when disabled
            return df

        # Step 1: Clean missing values
        df = self.clean_missing_values(df)

        # Step 2: Clean diagnosis codes
        df = self.clean_diagnosis_codes(df)

        # Step 3: Remove duplicates
        df = self.remove_duplicates(df)

        self.is_fitted = True
        self.logger.info("Preprocessing pipeline completed successfully")

        return df


def preprocess_diabetic_data(config_path: str = "config.yaml") -> pd.DataFrame:
    """
    Convenience function to preprocess diabetic readmission data.

    Args:
        config_path (str): Path to configuration file

    Returns:
        pd.DataFrame: Preprocessed data
    """
    preprocessor = DiabetesDataPreprocessor(config_path)
    df = preprocessor.load_data()
    return preprocessor.fit_transform(df)


if __name__ == "__main__":
    """
    CLI entry point for preprocessing data.

    Usage:
        python -m src.preprocess.preprocessing <config.yaml>
        or
        python -m src.preprocess.preprocessing <data.csv> <config.yaml>
    """
    logger = setup_logging()

    if len(sys.argv) == 2:
        # Single argument: config file path, load data from config
        config_path = sys.argv[1]
        logger.info(f"Preprocessing data using config: {config_path}")

        try:
            df = preprocess_diabetic_data(config_path)
            logger.info(f"Preprocessing completed. Final shape: {df.shape}")
            logger.info(f"Columns: {list(df.columns)}")
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            sys.exit(1)

    elif len(sys.argv) == 3:
        # Two arguments: data file and config file
        data_path, config_path = sys.argv[1], sys.argv[2]
        logger.info(f"Preprocessing data file: {data_path}\
             with config: {config_path}")

        try:
            preprocessor = DiabetesDataPreprocessor(config_path)
            df = preprocessor.load_data(data_path)
            df_processed = preprocessor.fit_transform(df)

            # Save processed data
            output_path = data_path.replace('.csv', '_preprocessed.csv')
            df_processed.to_csv(output_path, index=False)
            logger.info(f"Processed data saved to: {output_path}")
            logger.info(f"Final shape: {df_processed.shape}")

        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            sys.exit(1)
    else:
        logger.error("Usage: python -m src.preprocess.preprocessing\
             <config.yaml> OR <data.csv> <config.yaml>")
        sys.exit(1)
