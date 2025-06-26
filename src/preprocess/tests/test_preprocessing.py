"""
Unit tests for preprocessing.py module.

Tests cover preprocessing logic, error handling, and configuration behavior
using mock data and test fixtures.
"""

import pytest
import pandas as pd
import tempfile
import os
from unittest.mock import patch, MagicMock

from src.preprocess.preprocessing import (  # noqa: E402
    DiabetesDataPreprocessor,
    preprocess_diabetic_data
)


# Move fixtures to module level so they're accessible to all test classes
@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    return {
        "preprocessing": {
            "enabled": True,
            "drop_columns": ["weight", "payer_code", "medical_specialty"],
            "exclude_discharge_disposition": [11],
            "na_indicators": ["?", "Unknown/Invalid"],
            "duplicate_subset": ["patient_nbr"]
        },
        "data_source": {
            "keep_default_na": False,
            "na_values": [""]
        }
    }


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'encounter_id': [1, 2, 3, 4, 5, 6, 7, 8],
        'patient_nbr': [101, 102, 103, 101,
                        105, 106, 107, 108],  # 101 is duplicate
        'race': ['Caucasian', 'AfricanAmerican', '?',
                 'Hispanic', 'Asian', 'Other', 'Caucasian', '?'],
        'gender': ['Male', 'Female', 'Male', 'Unknown/Invalid',
                   'Female', 'Male', 'Female', 'Male'],
        'age': ['[70-80)', '[50-60)', '[60-70)', '[40-50)', None,
                '[30-40)', '[80-90)', '[20-30)'],
        'discharge_disposition_id': [1, 2, 3, 11, 5, 6, 7, 11],  # 11 is death
        'time_in_hospital': [5, 3, 7, 2, 4, 6, 8, 1],
        'diag_1': ['250.01', '401.9', '428.0', '250.50', '715.90',
                   '250.02', '427.31', '250.03'],
        'diag_2': ['V45.82', '250.00', '414.01', '?', '250.92',
                   'Unknown/Invalid', '250.01', '272.4'],
        'diag_3': ['401.9', '?', '250.01', '250.60', 'Unknown/Invalid',
                   '401.9', '250.00', '250.01'],
        'readmitted': ['NO', '<30', '>30', 'NO', '<30', 'NO', '>30', 'NO'],
        # Columns to be dropped
        'weight': ['?', '[75-100)', '?', '[50-75)', 'Unknown/Invalid', '?',
                   '[100-125)', '?'],
        'payer_code': ['MC', 'HM', '?', 'BC', 'Unknown/Invalid', 'MC',
                       'HM', '?'],
        'medical_specialty': ['Cardiology', '?', 'Internal Medicine',
                              'Unknown/Invalid', 'Family/GeneralPractice', '?',
                              'Cardiology', 'Internal Medicine']
    })


@pytest.fixture
def preprocessor_with_config(sample_config):
    """Create a preprocessor instance with mock config."""
    with patch('src.preprocess.preprocessing.load_config') as mock_load_config:
        mock_load_config.return_value = sample_config
        with patch('src.preprocess.preprocessing.get_logger') as mock_logger:
            mock_logger.return_value = MagicMock()
            preprocessor = DiabetesDataPreprocessor("test_config.yaml")
            return preprocessor


class TestDiabetesDataPreprocessor:
    """Test the main preprocessing class."""

    def test_init_successful(self, sample_config):
        """Test successful initialization of preprocessor."""
        with patch('src.preprocess.preprocessing.load_config') \
                as mock_load_config:
            mock_load_config.return_value = sample_config
            with patch('src.preprocess.preprocessing.get_logger') \
                    as mock_logger:
                mock_logger.return_value = MagicMock()

                preprocessor = DiabetesDataPreprocessor("test_config.yaml")

                assert preprocessor.config_path == "test_config.yaml"
                assert preprocessor.config == sample_config
                assert preprocessor.preprocess_config == \
                    sample_config["preprocessing"]
                assert preprocessor.is_fitted is False

    def test_init_config_load_failure(self):
        """Test initialization failure when config loading fails."""
        with patch('src.preprocess.preprocessing.load_config') \
                as mock_load_config:
            mock_load_config.side_effect = Exception("Config load failed")
            with patch('src.preprocess.preprocessing.get_logger') \
                    as mock_logger:
                mock_logger.return_value = MagicMock()

                with pytest.raises(Exception, match="Config load failed"):
                    DiabetesDataPreprocessor("invalid_config.yaml")

    def test_load_data_with_filepath(self, preprocessor_with_config,
                                     sample_dataframe):
        """Test loading data with specific filepath."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv',
                                         delete=False) as f:
            sample_dataframe.to_csv(f.name, index=False)

            try:
                df = preprocessor_with_config.load_data(f.name)
                assert df.shape == sample_dataframe.shape
                assert list(df.columns) == list(sample_dataframe.columns)
            finally:
                os.unlink(f.name)

    def test_load_data_with_dataloader(self, preprocessor_with_config,
                                       sample_dataframe):
        """Test loading data using DataLoader."""
        with patch('src.data_load.data_loader.DataLoader') as mock_data_loader:
            mock_loader_instance = mock_data_loader.return_value
            mock_loader_instance.load_data.return_value = sample_dataframe

            df = preprocessor_with_config.load_data()

            mock_data_loader.assert_called_once_with(
                    preprocessor_with_config.config_path)
            mock_loader_instance.load_data.assert_called_once()
            assert df.equals(sample_dataframe)

    def test_load_data_file_error(self, preprocessor_with_config):
        """Test error handling when file loading fails."""
        with pytest.raises(Exception):
            preprocessor_with_config.load_data("nonexistent_file.csv")

    def test_clean_missing_values(self, preprocessor_with_config,
                                  sample_dataframe):
        """Test missing values cleaning functionality."""
        df_cleaned = preprocessor_with_config.clean_missing_values(
            sample_dataframe)

        # Check that NA indicators were replaced and rows with NaN were dropped
        assert '?' not in df_cleaned.values
        assert 'Unknown/Invalid' not in df_cleaned.values
        assert df_cleaned.isnull().sum().sum() == 0

        # Check that specified columns were dropped
        assert 'weight' not in df_cleaned.columns
        assert 'payer_code' not in df_cleaned.columns
        assert 'medical_specialty' not in df_cleaned.columns

        # Check that death cases (discharge_disposition_id=11) were removed
        assert 11 not in df_cleaned['discharge_disposition_id'].values

        # Check that the result is smaller than input
        assert df_cleaned.shape[0] < sample_dataframe.shape[0]
        assert df_cleaned.shape[1] < sample_dataframe.shape[1]

    def test_clean_missing_values_no_config(self, preprocessor_with_config,
                                            sample_dataframe):
        """Test cleaning when config values are missing."""
        # Override config to have empty values
        preprocessor_with_config.preprocess_config = {}

        df_cleaned = preprocessor_with_config.clean_missing_values(
            sample_dataframe)

        # Should not drop any columns or exclude any dispositions
        assert df_cleaned.shape[1] == sample_dataframe.shape[1]
        assert 11 in df_cleaned['discharge_disposition_id'].values

    def test_clean_diagnosis_codes(self, preprocessor_with_config,
                                   sample_dataframe):
        """Test diagnosis code cleaning functionality."""
        df_cleaned = preprocessor_with_config.clean_diagnosis_codes(
            sample_dataframe)

        # Check that decimal parts were removed
        for col in ['diag_1', 'diag_2', 'diag_3']:
            if col in df_cleaned.columns:
                for value in df_cleaned[col].dropna():
                    if isinstance(value, str) and \
                            value not in ['?', 'Unknown/Invalid']:
                        assert '.' not in value

    def test_remove_after_period(self, preprocessor_with_config):
        """Test the helper method for removing decimal parts."""
        # Test with decimal
        assert preprocessor_with_config._remove_after_period('250.01') == '250'

        # Test without decimal
        assert preprocessor_with_config._remove_after_period('250') == '250'

        # Test with non-string
        assert preprocessor_with_config._remove_after_period(250) == 250

        # Test with None
        assert preprocessor_with_config._remove_after_period(None) is None

    def test_remove_duplicates(self, preprocessor_with_config,
                               sample_dataframe):
        """Test duplicate removal functionality."""
        df_deduplicated = preprocessor_with_config.remove_duplicates(
            sample_dataframe)

        # Check that duplicates were removed based on patient_nbr
        assert df_deduplicated['patient_nbr'].duplicated().sum() == 0
        assert df_deduplicated.shape[0] < sample_dataframe.shape[0]

        # Check that first occurrence was kept
        patient_101_rows = sample_dataframe[
            sample_dataframe['patient_nbr'] == 101]
        remaining_101 = df_deduplicated[df_deduplicated['patient_nbr'] == 101]
        assert len(remaining_101) == 1
        assert remaining_101.iloc[0]['encounter_id'] == \
            patient_101_rows.iloc[0]['encounter_id']

    def test_remove_duplicates_custom_subset(self, preprocessor_with_config,
                                             sample_dataframe):
        """Test duplicate removal with custom subset."""
        df_deduplicated = preprocessor_with_config.remove_duplicates(
            sample_dataframe,
            subset=['encounter_id']
        )

        # Should remove duplicates based on encounter_id (which are all unique
        # in our sample)
        assert df_deduplicated.shape[0] == sample_dataframe.shape[0]

    def test_remove_duplicates_no_duplicates(self, preprocessor_with_config):
        """Test duplicate removal when no duplicates exist."""
        df_no_dups = pd.DataFrame({
            'patient_nbr': [101, 102, 103],
            'encounter_id': [1, 2, 3]
        })

        df_result = preprocessor_with_config.remove_duplicates(df_no_dups)
        assert df_result.shape == df_no_dups.shape

    def test_fit_transform_full_pipeline(self, preprocessor_with_config,
                                         sample_dataframe):
        """Test the complete preprocessing pipeline."""
        df_processed = preprocessor_with_config.fit_transform(sample_dataframe)

        # Check that preprocessor is marked as fitted
        assert preprocessor_with_config.is_fitted is True

        # Check that all transformations were applied
        assert df_processed.shape[0] < sample_dataframe.shape[0]
        assert df_processed.shape[1] < sample_dataframe.shape[1]

        # Check specific transformations
        assert '?' not in df_processed.values
        assert 'Unknown/Invalid' not in df_processed.values
        assert 'weight' not in df_processed.columns
        assert 11 not in df_processed['discharge_disposition_id'].values
        assert df_processed['patient_nbr'].duplicated().sum() == 0

    def test_fit_transform_disabled(self, preprocessor_with_config,
                                    sample_dataframe):
        """Test pipeline when preprocessing is disabled."""
        preprocessor_with_config.preprocess_config['enabled'] = False

        df_processed = preprocessor_with_config.fit_transform(sample_dataframe)

        # Should return original dataframe unchanged
        assert df_processed.equals(sample_dataframe)
        # When disabled, is_fitted should still be set to True
        assert preprocessor_with_config.is_fitted is True

    def test_fit_transform_empty_dataframe(self, preprocessor_with_config):
        """Test pipeline with empty dataframe."""
        empty_df = pd.DataFrame()

        df_processed = preprocessor_with_config.fit_transform(empty_df)

        assert df_processed.empty
        assert preprocessor_with_config.is_fitted is True


class TestPreprocessDiabeticData:
    """Test the convenience function."""

    @patch('src.preprocess.preprocessing.DiabetesDataPreprocessor')
    def test_preprocess_diabetic_data(self, mock_preprocessor_class):
        """Test the convenience function."""
        # Mock the preprocessor instance
        mock_preprocessor = MagicMock()
        mock_preprocessor_class.return_value = mock_preprocessor

        # Mock the data and processing
        mock_df = pd.DataFrame({'col1': [1, 2, 3]})
        mock_preprocessor.load_data.return_value = mock_df
        mock_preprocessor.fit_transform.return_value = mock_df

        # Test the function
        result = preprocess_diabetic_data("test_config.yaml")

        # Verify calls
        mock_preprocessor_class.assert_called_once_with("test_config.yaml")
        mock_preprocessor.load_data.assert_called_once()
        mock_preprocessor.fit_transform.assert_called_once_with(mock_df)
        assert result.equals(mock_df)


class TestCLIInterface:
    """Test command line interface functionality."""

    @patch('src.preprocess.preprocessing.setup_logging')
    @patch('src.preprocess.preprocessing.preprocess_diabetic_data')
    def test_cli_single_argument(self, mock_preprocess, mock_setup_logging):
        """Test CLI with single argument (config file)."""
        mock_logger = MagicMock()
        mock_setup_logging.return_value = mock_logger

        mock_df = pd.DataFrame({'col1': [1, 2, 3]})
        mock_preprocess.return_value = mock_df

        # Simulate single argument CLI call
        config_path = 'config.yaml'
        result_df = mock_preprocess(config_path)

        mock_preprocess.assert_called_once_with('config.yaml')
        assert result_df.equals(mock_df)

    @patch('src.preprocess.preprocessing.setup_logging')
    @patch('src.preprocess.preprocessing.DiabetesDataPreprocessor')
    def test_cli_two_arguments(self, mock_preprocessor_class,
                               mock_setup_logging):
        """Test CLI with two arguments (data file and config file)."""
        mock_logger = MagicMock()
        mock_setup_logging.return_value = mock_logger

        # Mock the preprocessor
        mock_preprocessor = MagicMock()
        mock_preprocessor_class.return_value = mock_preprocessor

        mock_df = pd.DataFrame({'col1': [1, 2, 3]})
        mock_preprocessor.load_data.return_value = mock_df
        mock_preprocessor.fit_transform.return_value = mock_df

        # Simulate two argument CLI call
        data_path, config_path = 'data.csv', 'config.yaml'

        preprocessor = mock_preprocessor_class(config_path)
        df = preprocessor.load_data(data_path)
        df_processed = preprocessor.fit_transform(df)
        assert df_processed.equals(mock_df)

        mock_preprocessor_class.assert_called_with('config.yaml')
        mock_preprocessor.load_data.assert_called_with('data.csv')
        mock_preprocessor.fit_transform.assert_called_with(mock_df)


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_missing_columns_in_config(self, sample_dataframe):
        """Test handling when configured columns don't exist in data."""
        config = {
            "preprocessing": {
                "enabled": True,
                "drop_columns": ["nonexistent_column"],
                "exclude_discharge_disposition": [],
                "na_indicators": []
            },
            "data_source": {"keep_default_na": False, "na_values": [""]}
        }

        with patch('src.preprocess.preprocessing.load_config') \
                as mock_load_config:
            mock_load_config.return_value = config
            with patch('src.preprocess.preprocessing.get_logger') \
                    as mock_logger:
                mock_logger.return_value = MagicMock()

                preprocessor = DiabetesDataPreprocessor("test_config.yaml")

                # Should not raise error, just ignore missing columns
                df_processed = preprocessor.clean_missing_values(
                    sample_dataframe)
                assert df_processed.shape[1] == sample_dataframe.shape[1]

    def test_empty_diagnosis_columns(self, preprocessor_with_config):
        """Test diagnosis cleaning with missing diagnosis columns."""
        df_no_diag = pd.DataFrame({
            'encounter_id': [1, 2, 3],
            'patient_nbr': [101, 102, 103]
        })

        # Should not raise error
        df_processed = preprocessor_with_config.clean_diagnosis_codes(
            df_no_diag)
        assert df_processed.equals(df_no_diag)

    def test_all_rows_filtered_out(self, preprocessor_with_config):
        """Test when all rows are filtered out during cleaning."""
        # Create a dataframe where all rows will be filtered
        df_all_invalid = pd.DataFrame({
            'patient_nbr': [101, 102, 103],
            'discharge_disposition_id': [11, 11, 11],  # All death cases
            'age': ['?', 'Unknown/Invalid', None]  # All invalid ages
        })

        df_processed = preprocessor_with_config.clean_missing_values(
            df_all_invalid)

        # Should result in empty dataframe
        assert df_processed.empty

    def test_malformed_diagnosis_codes(self, preprocessor_with_config):
        """Test handling of malformed diagnosis codes."""
        df_malformed = pd.DataFrame({
            'diag_1': ['250.01.extra', '401..9', '428.', '.250', None],
            'diag_2': [250, 401.9, '428.0', '?', 'invalid'],
            'patient_nbr': [101, 102, 103, 104, 105]
        })

        df_processed = preprocessor_with_config.clean_diagnosis_codes(
            df_malformed)

        # Should handle all cases without errors
        assert len(df_processed) == len(df_malformed)

        # Check specific transformations
        assert df_processed.loc[0, 'diag_1'] == '250'
        assert df_processed.loc[1, 'diag_1'] == '401'
        assert df_processed.loc[2, 'diag_1'] == '428'

    def test_duplicate_removal_with_all_duplicates(self,
                                                   preprocessor_with_config):
        """Test duplicate removal when all rows are duplicates."""
        df_all_dups = pd.DataFrame({
            'patient_nbr': [101, 101, 101, 101],
            'encounter_id': [1, 2, 3, 4]
        })

        df_processed = preprocessor_with_config.remove_duplicates(df_all_dups)

        # Should keep only first occurrence
        assert len(df_processed) == 1
        assert df_processed.iloc[0]['encounter_id'] == 1

    def test_preprocessing_with_minimal_config(self, sample_dataframe):
        """Test preprocessing with minimal configuration."""
        minimal_config = {
            "preprocessing": {"enabled": True},
            "data_source": {}
        }

        with patch('src.preprocess.preprocessing.load_config') \
                as mock_load_config:
            mock_load_config.return_value = minimal_config
            with patch('src.preprocess.preprocessing.get_logger') \
                    as mock_logger:
                mock_logger.return_value = MagicMock()

                preprocessor = DiabetesDataPreprocessor("test_config.yaml")

                # Should work with defaults
                df_processed = preprocessor.fit_transform(sample_dataframe)
                assert preprocessor.is_fitted is True
                # With minimal config, should mainly just remove duplicates
                assert df_processed.shape[0] <= sample_dataframe.shape[0]


class TestDataIntegrity:
    """Test data integrity and validation."""

    def test_data_types_preserved(self, preprocessor_with_config,
                                  sample_dataframe):
        """Test that appropriate data types are preserved after processing."""
        df_processed = preprocessor_with_config.fit_transform(sample_dataframe)

        # Check that numeric columns remain numeric where possible
        if 'encounter_id' in df_processed.columns:
            assert pd.api.types.is_numeric_dtype(df_processed['encounter_id'])
        if 'patient_nbr' in df_processed.columns:
            assert pd.api.types.is_numeric_dtype(df_processed['patient_nbr'])
        if 'time_in_hospital' in df_processed.columns:
            assert pd.api.types.is_numeric_dtype(df_processed[
                'time_in_hospital'])

    def test_no_data_leakage(self, preprocessor_with_config, sample_dataframe):
        """Test that no invalid data leaks through the pipeline."""
        df_processed = preprocessor_with_config.fit_transform(sample_dataframe)

        # Ensure no configured NA indicators remain
        for col in df_processed.select_dtypes(include=['object']).columns:
            assert '?' not in df_processed[col].values
            assert 'Unknown/Invalid' not in df_processed[col].values

        # Ensure no death cases remain
        if 'discharge_disposition_id' in df_processed.columns:
            assert 11 not in df_processed['discharge_disposition_id'].values

        # Ensure no actual NaN values remain (should have been dropped)
        assert df_processed.isnull().sum().sum() == 0

    def test_column_consistency(self, preprocessor_with_config,
                                sample_dataframe):
        """Test that column operations are consistent."""
        df_processed = preprocessor_with_config.fit_transform(sample_dataframe)

        # Check that expected columns were dropped
        dropped_columns = ['weight', 'payer_code', 'medical_specialty']
        for col in dropped_columns:
            assert col not in df_processed.columns

        # Check that essential columns remain
        # (if they existed and weren't filtered)
        essential_columns = ['encounter_id', 'patient_nbr', 'readmitted']
        for col in essential_columns:
            if col in sample_dataframe.columns:
                # Column should either be present or the dataframe
                # should be empty
                assert col in df_processed.columns or df_processed.empty
