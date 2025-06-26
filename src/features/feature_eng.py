"""
feature_eng.py

Feature engineering pipeline for diabetes readmission prediction.
Handles ICD-9 categorization, medication features, and value mappings.
Integrates with the MLOps configuration system for consistency.
"""

import pandas as pd
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.base import BaseEstimator, TransformerMixin

# Add project root to Python path before importing from src
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Now import from src
from src.utils.config import load_config  # noqa: E402
from src.utils.logging import setup_logging, get_logger  # noqa: E402

logger = get_logger(__name__)

pd.set_option('future.no_silent_downcasting', True)


class ICD9Categorizer(BaseEstimator, TransformerMixin):
    """
    Categorize ICD-9 diagnosis codes into disease groups.

    Clinical motivation:
    - Groups related diagnoses for better feature representation
    - Reduces dimensionality while preserving clinical meaning
    - Improves model interpretability for healthcare professionals

    Usage:
        categorizer = ICD9Categorizer(icd9_categories)
        df_transformed = categorizer.fit_transform(df)
    """

    def __init__(self, icd9_categories: Dict[str, List[Tuple[int, int]]]):
        self.icd9_categories = icd9_categories

    def fit(self, X, y=None):
        # No fitting necessary for rule-based categorization
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        for col in ['diag_1', 'diag_2', 'diag_3']:
            if col in X.columns:
                X[col] = X[col].apply(
                    lambda x: self._categorize_icd9(x, self.icd9_categories)
                )

        return X

    def _categorize_icd9(self, code, categories):
        """Categorize a single ICD-9 code."""
        if pd.isna(code):
            return 'Other'

        code_str = str(code)
        if code_str.startswith('E') or code_str.startswith('V'):
            return 'Other'

        try:
            code_num = int(code_str)
            for category, ranges in categories.items():
                for start, end in ranges:
                    if start <= code_num <= end:
                        return category
        except ValueError:
            pass

        return 'Other'


class MedicationFeatureCreator(BaseEstimator, TransformerMixin):
    """
    Create medication-related features.

    Clinical motivation:
    - Medication changes indicate treatment intensity and complexity
    - Number of medications reflects diabetes severity
    - These features are strong predictors of readmission risk

    Usage:
        med_creator = MedicationFeatureCreator(medication_columns)
        df_transformed = med_creator.fit_transform(df)
    """

    def __init__(self, medication_columns: List[str]):
        self.medication_columns = medication_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # Count medication changes (before applying mappings)
        X['numchange'] = 0
        for col in self.medication_columns:
            if col in X.columns:
                X['numchange'] += (~X[col].isin(['No', 'Steady'])).astype(int)

        return X


class ValueMapper(BaseEstimator, TransformerMixin):
    """
    Apply value mappings to convert categorical to numeric variables.

    Clinical motivation:
    - Converts categorical variables to meaningful numeric representations
    - Maintains clinical interpretation while enabling ML algorithms
    - Handles ordinal relationships in age and test results

    Usage:
        mapper = ValueMapper(value_mappings)
        df_transformed = mapper.fit_transform(df)
    """

    def __init__(self, value_mappings: Dict):
        self.value_mappings = value_mappings

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # Apply ID mappings
        for col in ['admission_type_id', 'discharge_disposition_id',
                    'admission_source_id']:
            if col in X.columns and col in self.value_mappings:
                mapping = self.value_mappings[col]
                X[col] = X[col].replace(mapping).astype('int64')

        # Apply binary variable mappings
        if 'binary_vars' in self.value_mappings:
            for col, mapping in self.value_mappings['binary_vars'].items():
                if col in X.columns:
                    X[col] = X[col].replace(mapping).astype('int64')

        # Apply lab test mappings
        if 'lab_tests' in self.value_mappings:
            for col, mapping in self.value_mappings['lab_tests'].items():
                if col in X.columns:
                    X[col] = X[col].replace(mapping).astype('int64')

        # Apply age mapping
        if 'age' in self.value_mappings and 'age' in X.columns:
            X['age'] = X['age'].replace(
                self.value_mappings['age']).astype('int64')

        # Apply readmission mapping
        if 'readmission' in self.value_mappings and 'readmitted' in X.columns:
            X['readmitted'] = X['readmitted'].replace(
                self.value_mappings['readmission']
            ).astype('int64')

        # Apply medication status mappings
        if 'medication_status' in self.value_mappings:
            med_mapping = self.value_mappings['medication_status']
            medication_cols = self.value_mappings.get('medication_columns', [])
            for col in medication_cols:
                if col in X.columns:
                    X[col] = X[col].replace(med_mapping).astype('int64')

        return X


class AdditionalFeatureCreator(BaseEstimator, TransformerMixin):
    """
    Create additional derived features.

    Clinical motivation:
    - Total medication count indicates treatment complexity
    - Derived features capture clinical concepts not explicit in raw data
    - These composite features often have stronger predictive power

    Usage:
        feature_creator = AdditionalFeatureCreator(medication_columns)
        df_transformed = feature_creator.fit_transform(df)
    """

    def __init__(self, medication_columns: List[str]):
        self.medication_columns = medication_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # Count total medications (after applying mappings)
        X['nummed'] = 0
        for col in self.medication_columns:
            if col in X.columns:
                # Ensure numeric conversion
                X['nummed'] += pd.to_numeric(
                    X[col], errors='coerce').fillna(0).astype(int)

        return X


class DiabetesFeatureEngineer:
    """
    Feature engineering pipeline for diabetes readmission prediction.
    Handles ICD-9 categorization, medication features, and value mappings.
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the feature engineer with configuration.

        Args:
            config_path (str): Path to the configuration file
        """
        self.logger = get_logger(__name__)
        self.config_path = config_path

        # Load configuration
        self.config = load_config(self.logger, config_path)
        self.feature_config = self.config.get("features", {})

        self.is_fitted = False
        self._setup_transformers()

    def _setup_transformers(self):
        """Initialize transformer objects."""
        medication_columns = self.feature_config.get('medication_columns', [])
        icd9_categories = self.feature_config.get('icd9_categories', {})
        value_mappings = self.feature_config.get('value_mappings', {})

        self.icd9_categorizer = ICD9Categorizer(icd9_categories)
        self.medication_creator = MedicationFeatureCreator(medication_columns)
        self.value_mapper = ValueMapper(value_mappings)
        self.additional_creator = AdditionalFeatureCreator(medication_columns)

    def load_data(self, filepath: str = None) -> pd.DataFrame:
        """Load data for feature engineering."""
        if filepath:
            # If specific filepath provided, load directly
            try:
                df = pd.read_csv(filepath)
                self.logger.info(
                    f"Data loaded successfully. Shape: {df.shape}")
                return df
            except Exception as e:
                self.logger.error(f"Error loading data: {e}")
                raise
        else:
            # Use preprocessed data from preprocessing step
            from src.preprocess.preprocessing \
                import DiabetesDataPreprocessor  # noqa: E402
            preprocessor = DiabetesDataPreprocessor(self.config_path)
            return preprocessor.fit_transform(preprocessor.load_data())

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Complete feature engineering pipeline."""
        self.logger.info("Starting feature engineering pipeline...")

        # Check if feature engineering is enabled
        if not self.feature_config.get('enabled', True):
            self.logger.info("Feature engineering is disabled in config.")
            self.is_fitted = True
            return df

        # Step 1: Categorize diagnosis codes
        self.logger.info("Categorizing ICD-9 diagnosis codes...")
        df = self.icd9_categorizer.fit_transform(df)

        # Step 2: Create medication features (before mappings)
        self.logger.info("Creating medication features...")
        df = self.medication_creator.fit_transform(df)

        # Step 3: Apply value mappings
        self.logger.info("Applying value mappings...")
        df = self.value_mapper.fit_transform(df)

        # Step 4: Create additional features (after mappings)
        self.logger.info("Creating additional derived features...")
        df = self.additional_creator.fit_transform(df)

        self.is_fitted = True
        self.logger.info("Feature engineering pipeline completed successfully")

        return df


# Feature transformer registry for easy access
FEATURE_TRANSFORMERS = {
    "icd9_categorizer": lambda config: ICD9Categorizer(config.get(
        "icd9_categories", {})),
    "medication_creator": lambda config: MedicationFeatureCreator(config.get(
        "medication_columns", [])),
    "value_mapper": lambda config: ValueMapper(config.get(
        "value_mappings", {})),
    "additional_creator": lambda config: AdditionalFeatureCreator(config.get(
        "medication_columns", []))
}


def engineer_diabetes_features(
        config_path: str = "config.yaml") -> pd.DataFrame:
    """
    Convenience function to engineer features for diabetic readmission data.

    Args:
        config_path (str): Path to configuration file

    Returns:
        pd.DataFrame: Feature-engineered data
    """
    engineer = DiabetesFeatureEngineer(config_path)
    df = engineer.load_data()
    return engineer.fit_transform(df)


if __name__ == "__main__":
    """
    CLI entry point for feature engineering.

    Usage:
        python -m src.features.feature_eng <config.yaml>
        or
        python -m src.features.feature_eng <data.csv> <config.yaml>
    """
    logger = setup_logging()

    if len(sys.argv) == 2:
        # Single argument: config file path, load data from preprocessing
        config_path = sys.argv[1]
        logger.info(f"Engineering features using config: {config_path}")

        try:
            df = engineer_diabetes_features(config_path)
            logger.info(f"Feature engineering completed.\
                 Final shape: {df.shape}")
            logger.info(f"Columns: {list(df.columns)}")
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            sys.exit(1)

    elif len(sys.argv) == 3:
        # Two arguments: data file and config file
        data_path, config_path = sys.argv[1], sys.argv[2]
        logger.info(f"Engineering features for data file: {data_path}\
             with config: {config_path}")

        try:
            engineer = DiabetesFeatureEngineer(config_path)
            df = engineer.load_data(data_path)
            df_engineered = engineer.fit_transform(df)

            # Save engineered data
            output_path = data_path.replace('.csv', '_engineered.csv')
            df_engineered.to_csv(output_path, index=False)
            logger.info(f"Engineered data saved to: {output_path}")
            logger.info(f"Final shape: {df_engineered.shape}")

        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            sys.exit(1)
    else:
        logger.error("Usage: python -m src.features.feature_eng <config.yaml>\
             OR <data.csv> <config.yaml>")
        sys.exit(1)
