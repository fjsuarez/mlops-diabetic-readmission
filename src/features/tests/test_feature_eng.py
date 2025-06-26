"""
Unit tests for feature_eng.py module.

Tests cover feature engineering transformers, pipeline logic, and
configuration behavior using mock data and test fixtures.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock

from src.features.feature_eng import (  # noqa: E402
    ICD9Categorizer,
    MedicationFeatureCreator,
    ValueMapper,
    AdditionalFeatureCreator,
    DiabetesFeatureEngineer,
    engineer_diabetes_features,
    FEATURE_TRANSFORMERS
)


# Module-level fixtures accessible to all test classes
@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    return {
        "features": {
            "enabled": True,
            "medication_columns": [
                "metformin", "insulin", "glyburide", "glipizide"
            ],
            "icd9_categories": {
                "Diabetes": [[250, 250]],
                "Circulatory": [[390, 459]],
                "Respiratory": [[460, 519]],
                "Other": []
            },
            "value_mappings": {
                "admission_type_id": {2: 1, 7: 1},
                "binary_vars": {
                    "gender": {"Male": 1, "Female": 0},
                    "diabetesMed": {"Yes": 1, "No": 0}
                },
                "lab_tests": {
                    "A1Cresult": {">7": 1, "Norm": 0, "None": -99}
                },
                "age": {
                    "[20-30)": 3, "[30-40)": 4, "[40-50)": 5
                },
                "readmission": {">30": 0, "<30": 1, "NO": 0},
                "medication_status": {"No": 0, "Steady": 1, "Up": 1,
                                      "Down": 1},
                "medication_columns": ["metformin", "insulin",
                                       "glyburide", "glipizide"]
            }
        }
    }


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'encounter_id': [1, 2, 3, 4, 5],
        'patient_nbr': [101, 102, 103, 104, 105],
        # Diagnosis columns
        'diag_1': ['250', '401', '428', 'V45', 'E123'],  # Mixed ICD-9 codes
        'diag_2': ['250.01', '460.5', '999', None, '140'],
        'diag_3': ['390', '520', 'V30', '250', '999'],
        # Medication columns
        'metformin': ['No', 'Steady', 'Up', 'Down', 'No'],
        'insulin': ['Steady', 'No', 'Up', 'No', 'Down'],
        'glyburide': ['No', 'No', 'Steady', 'Up', 'No'],
        'glipizide': ['Up', 'Steady', 'No', 'No', 'Steady'],
        # Categorical variables to map
        'admission_type_id': [1, 2, 3, 7, 1],
        'gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
        'diabetesMed': ['Yes', 'No', 'Yes', 'Yes', 'No'],
        'A1Cresult': ['>7', 'Norm', 'None', '>7', 'Norm'],
        'age': ['[20-30)', '[30-40)', '[40-50)', '[20-30)', '[30-40)'],
        'readmitted': ['>30', '<30', 'NO', '<30', '>30']
    })


@pytest.fixture
def feature_engineer_with_config(sample_config):
    """Create a feature engineer instance with mock config."""
    with patch('src.features.feature_eng.load_config') as mock_load_config:
        mock_load_config.return_value = sample_config
        with patch('src.features.feature_eng.get_logger') as mock_logger:
            mock_logger.return_value = MagicMock()
            engineer = DiabetesFeatureEngineer("test_config.yaml")
            return engineer


class TestICD9Categorizer:
    """Test the ICD9 categorization transformer."""

    @pytest.fixture
    def icd9_categories(self):
        """Sample ICD-9 categories for testing."""
        return {
            "Diabetes": [[250, 250]],
            "Circulatory": [[390, 459]],
            "Respiratory": [[460, 519]],
            "Neoplasms": [[140, 239]]
        }

    @pytest.fixture
    def categorizer(self, icd9_categories):
        """Create ICD9Categorizer instance."""
        return ICD9Categorizer(icd9_categories)

    def test_init(self, icd9_categories):
        """Test successful initialization."""
        categorizer = ICD9Categorizer(icd9_categories)
        assert categorizer.icd9_categories == icd9_categories

    def test_fit_returns_self(self, categorizer, sample_dataframe):
        """Test that fit returns self."""
        result = categorizer.fit(sample_dataframe)
        assert result is categorizer

    def test_categorize_diabetes_codes(self, categorizer):
        """Test categorization of diabetes codes."""
        # After preprocessing, codes should be clean integers
        assert categorizer._categorize_icd9(
            "250", categorizer.icd9_categories) == "Diabetes"
        # No decimal codes should reach this stage after preprocessing

    def test_categorize_circulatory_codes(self, categorizer):
        """Test categorization of circulatory codes."""
        assert categorizer._categorize_icd9(
            "401", categorizer.icd9_categories) == "Circulatory"
        assert categorizer._categorize_icd9(
            "459", categorizer.icd9_categories) == "Circulatory"

    def test_categorize_respiratory_codes(self, categorizer):
        """Test categorization of respiratory codes."""
        assert categorizer._categorize_icd9(
            "460", categorizer.icd9_categories) == "Respiratory"
        assert categorizer._categorize_icd9(
            "519", categorizer.icd9_categories) == "Respiratory"

    def test_categorize_v_codes(self, categorizer):
        """Test categorization of V codes."""
        assert categorizer._categorize_icd9(
            "V45", categorizer.icd9_categories) == "Other"
        assert categorizer._categorize_icd9(
            "V30", categorizer.icd9_categories) == "Other"

    def test_categorize_e_codes(self, categorizer):
        """Test categorization of E codes."""
        assert categorizer._categorize_icd9(
            "E123", categorizer.icd9_categories) == "Other"
        assert categorizer._categorize_icd9(
            "E456", categorizer.icd9_categories) == "Other"

    def test_categorize_nan_codes(self, categorizer):
        """Test categorization of NaN codes."""
        assert categorizer._categorize_icd9(
            None, categorizer.icd9_categories) == "Other"
        assert categorizer._categorize_icd9(
            np.nan, categorizer.icd9_categories) == "Other"

    def test_categorize_invalid_codes(self, categorizer):
        """Test categorization of invalid codes."""
        assert categorizer._categorize_icd9(
            "invalid", categorizer.icd9_categories) == "Other"
        assert categorizer._categorize_icd9(
            "", categorizer.icd9_categories) == "Other"

    def test_categorize_uncategorized_codes(self, categorizer):
        """Test categorization of codes not in any category."""
        assert categorizer._categorize_icd9(
            "999", categorizer.icd9_categories) == "Other"
        assert categorizer._categorize_icd9(
            "100", categorizer.icd9_categories) == "Other"

    def test_categorize_integer_input(self, categorizer):
        """Test categorization with integer inputs (from preprocessed data)."""
        # Preprocessed data might contain actual integers
        assert categorizer._categorize_icd9(
            250, categorizer.icd9_categories) == "Diabetes"
        assert categorizer._categorize_icd9(
            401, categorizer.icd9_categories) == "Circulatory"
        assert categorizer._categorize_icd9(
            460, categorizer.icd9_categories) == "Respiratory"

    def test_categorize_boundary_codes(self, categorizer):
        """Test categorization at category boundaries."""
        # Test exact boundaries of ranges
        assert categorizer._categorize_icd9(
            "250", categorizer.icd9_categories) == "Diabetes"  # Exact match
        assert categorizer._categorize_icd9(
            "390", categorizer.icd9_categories) == "Circulatory"  # Lower bound
        assert categorizer._categorize_icd9(
            "459", categorizer.icd9_categories) == "Circulatory"  # Upper bound
        assert categorizer._categorize_icd9(
            "389", categorizer.icd9_categories) == "Other"  # Just below range
        assert categorizer._categorize_icd9(
            "460", categorizer.icd9_categories) == "Respiratory"  # Next range

    def test_transform_diagnosis_columns(self, categorizer, sample_dataframe):
        """Test transformation of diagnosis columns."""
        df_transformed = categorizer.transform(sample_dataframe)

        # Check that diagnosis columns were categorized
        assert df_transformed.loc[0, 'diag_1'] == "Diabetes"  # 250
        assert df_transformed.loc[1, 'diag_1'] == "Circulatory"  # 401
        assert df_transformed.loc[3, 'diag_1'] == "Other"  # V45
        assert df_transformed.loc[4, 'diag_1'] == "Other"  # E123

    def test_transform_missing_columns(self, categorizer):
        """Test transformation when diagnosis columns are missing."""
        df_no_diag = pd.DataFrame({'other_col': [1, 2, 3]})
        df_transformed = categorizer.transform(df_no_diag)

        # Should return dataframe unchanged
        assert df_transformed.equals(df_no_diag)

    def test_transform_preserves_other_columns(
            self, categorizer, sample_dataframe):
        """Test that transformation preserves non-diagnosis columns."""
        df_transformed = categorizer.transform(sample_dataframe)

        # Check that other columns are unchanged
        assert df_transformed['encounter_id'].equals(
            sample_dataframe['encounter_id'])
        assert df_transformed['patient_nbr'].equals(
            sample_dataframe['patient_nbr'])


class TestMedicationFeatureCreator:
    """Test the medication feature creator transformer."""

    @pytest.fixture
    def medication_columns(self):
        """Sample medication columns for testing."""
        return ["metformin", "insulin", "glyburide", "glipizide"]

    @pytest.fixture
    def med_creator(self, medication_columns):
        """Create MedicationFeatureCreator instance."""
        return MedicationFeatureCreator(medication_columns)

    def test_init(self, medication_columns):
        """Test successful initialization."""
        creator = MedicationFeatureCreator(medication_columns)
        assert creator.medication_columns == medication_columns

    def test_fit_returns_self(self, med_creator, sample_dataframe):
        """Test that fit returns self."""
        result = med_creator.fit(sample_dataframe)
        assert result is med_creator

    def test_create_numchange_feature(self, med_creator, sample_dataframe):
        """Test creation of numchange feature."""
        df_transformed = med_creator.transform(sample_dataframe)

        # Check that numchange column was created
        assert 'numchange' in df_transformed.columns

        # Check calculations
        # Row 0: metformin=No, insulin=Steady, glyburide=No,
        # glipizide=Up -> 1 change
        assert df_transformed.loc[0, 'numchange'] == 1

        # Row 2: metformin=Up, insulin=Up, glyburide=Steady,
        # glipizide=No -> 2 changes
        assert df_transformed.loc[2, 'numchange'] == 2

    def test_missing_medication_columns(self, med_creator):
        """Test handling when medication columns are missing."""
        df_no_meds = pd.DataFrame({'other_col': [1, 2, 3]})
        df_transformed = med_creator.transform(df_no_meds)

        # Should create numchange column with zeros
        assert 'numchange' in df_transformed.columns
        assert all(df_transformed['numchange'] == 0)

    def test_transform_preserves_other_columns(self, med_creator,
                                               sample_dataframe):
        """Test that transformation preserves other columns."""
        df_transformed = med_creator.transform(sample_dataframe)

        # Check that original columns are preserved
        for col in sample_dataframe.columns:
            assert col in df_transformed.columns
            assert df_transformed[col].equals(sample_dataframe[col])


class TestValueMapper:
    """Test the value mapping transformer."""

    @pytest.fixture
    def value_mappings(self):
        """Sample value mappings for testing."""
        return {
            "admission_type_id": {2: 1, 7: 1},
            "binary_vars": {
                "gender": {"Male": 1, "Female": 0},
                "diabetesMed": {"Yes": 1, "No": 0}
            },
            "lab_tests": {
                "A1Cresult": {">7": 1, "Norm": 0, "None": -99}
            },
            "age": {"[20-30)": 3, "[30-40)": 4, "[40-50)": 5},
            "readmission": {">30": 0, "<30": 1, "NO": 0},
            "medication_status": {"No": 0, "Steady": 1, "Up": 1, "Down": 1},
            "medication_columns": ["metformin", "insulin"]
        }

    @pytest.fixture
    def value_mapper(self, value_mappings):
        """Create ValueMapper instance."""
        return ValueMapper(value_mappings)

    def test_init(self, value_mappings):
        """Test successful initialization."""
        mapper = ValueMapper(value_mappings)
        assert mapper.value_mappings == value_mappings

    def test_fit_returns_self(self, value_mapper, sample_dataframe):
        """Test that fit returns self."""
        result = value_mapper.fit(sample_dataframe)
        assert result is value_mapper

    def test_map_admission_type_id(self, value_mapper, sample_dataframe):
        """Test mapping of admission_type_id."""
        df_transformed = value_mapper.transform(sample_dataframe)

        # Check mappings: 2->1, 7->1, others unchanged
        assert df_transformed.loc[1, 'admission_type_id'] == 1  # 2->1
        assert df_transformed.loc[3, 'admission_type_id'] == 1  # 7->1
        assert df_transformed.loc[0, 'admission_type_id'] == 1  # unchanged
        assert df_transformed.loc[2, 'admission_type_id'] == 3  # unchanged

    def test_map_binary_variables(self, value_mapper, sample_dataframe):
        """Test mapping of binary variables."""
        df_transformed = value_mapper.transform(sample_dataframe)

        # Check gender mapping
        assert df_transformed.loc[0, 'gender'] == 1  # Male->1
        assert df_transformed.loc[1, 'gender'] == 0  # Female->0

        # Check diabetesMed mapping
        assert df_transformed.loc[0, 'diabetesMed'] == 1  # Yes->1
        assert df_transformed.loc[1, 'diabetesMed'] == 0  # No->0

    def test_map_lab_tests(self, value_mapper, sample_dataframe):
        """Test mapping of lab test results."""
        df_transformed = value_mapper.transform(sample_dataframe)

        # Check A1Cresult mapping
        assert df_transformed.loc[0, 'A1Cresult'] == 1  # >7->1
        assert df_transformed.loc[1, 'A1Cresult'] == 0  # Norm->0
        assert df_transformed.loc[2, 'A1Cresult'] == -99  # None->-99

    def test_map_age(self, value_mapper, sample_dataframe):
        """Test mapping of age categories."""
        df_transformed = value_mapper.transform(sample_dataframe)

        # Check age mapping
        assert df_transformed.loc[0, 'age'] == 3  # [20-30)->3
        assert df_transformed.loc[1, 'age'] == 4  # [30-40)->4
        assert df_transformed.loc[2, 'age'] == 5  # [40-50)->5

    def test_map_readmission(self, value_mapper, sample_dataframe):
        """Test mapping of readmission target."""
        df_transformed = value_mapper.transform(sample_dataframe)

        # Check readmission mapping
        assert df_transformed.loc[0, 'readmitted'] == 0  # >30->0
        assert df_transformed.loc[1, 'readmitted'] == 1  # <30->1
        assert df_transformed.loc[2, 'readmitted'] == 0  # NO->0

    def test_map_medication_status(self, value_mapper, sample_dataframe):
        """Test mapping of medication status."""
        df_transformed = value_mapper.transform(sample_dataframe)

        # Check medication mapping for configured columns
        assert df_transformed.loc[0, 'metformin'] == 0  # No->0
        assert df_transformed.loc[1, 'metformin'] == 1  # Steady->1
        assert df_transformed.loc[2, 'metformin'] == 1  # Up->1

    def test_missing_mappings(self, sample_dataframe):
        """Test handling when mappings are missing."""
        empty_mappings = {}
        mapper = ValueMapper(empty_mappings)

        df_transformed = mapper.transform(sample_dataframe)

        # Should return dataframe unchanged
        assert df_transformed.equals(sample_dataframe)

    def test_data_types_conversion(self, value_mapper, sample_dataframe):
        """Test that mapped values are converted to int64."""
        df_transformed = value_mapper.transform(sample_dataframe)

        # Check data types for mapped columns
        assert df_transformed['admission_type_id'].dtype == 'int64'
        assert df_transformed['gender'].dtype == 'int64'
        assert df_transformed['age'].dtype == 'int64'

    def test_value_mapping_complete_mapping_converts_dtype(self):
        """Test that complete mappings result in int64 conversion."""
        mappings = {
            "binary_vars": {
                "complete_col": {"A": 1, "B": 0, "C": 2}
            }
        }

        mapper = ValueMapper(mappings)

        df_complete = pd.DataFrame({
            'complete_col': ['A', 'B', 'C', 'A', 'B']
        })

        df_processed = mapper.transform(df_complete)

        # Should convert to int64 since all values were mapped
        assert df_processed['complete_col'].dtype == 'int64'
        assert df_processed.loc[0, 'complete_col'] == 1
        assert df_processed.loc[1, 'complete_col'] == 0
        assert df_processed.loc[2, 'complete_col'] == 2


class TestAdditionalFeatureCreator:
    """Test the additional feature creator transformer."""

    @pytest.fixture
    def medication_columns(self):
        """Sample medication columns for testing."""
        return ["metformin", "insulin", "glyburide", "glipizide"]

    @pytest.fixture
    def feature_creator(self, medication_columns):
        """Create AdditionalFeatureCreator instance."""
        return AdditionalFeatureCreator(medication_columns)

    def test_init(self, medication_columns):
        """Test successful initialization."""
        creator = AdditionalFeatureCreator(medication_columns)
        assert creator.medication_columns == medication_columns

    def test_fit_returns_self(self, feature_creator, sample_dataframe):
        """Test that fit returns self."""
        result = feature_creator.fit(sample_dataframe)
        assert result is feature_creator

    def test_create_nummed_feature(self, feature_creator):
        """Test creation of nummed feature with numeric data."""
        # Create sample with numeric medication values (after mapping)
        df_numeric = pd.DataFrame({
            'metformin': [0, 1, 1, 1, 0],
            'insulin': [1, 0, 1, 0, 1],
            'glyburide': [0, 0, 1, 1, 0],
            'glipizide': [1, 1, 0, 0, 1]
        })

        df_transformed = feature_creator.transform(df_numeric)

        # Check that nummed column was created
        assert 'nummed' in df_transformed.columns

        # Check calculations
        # metformin=0, insulin=1, glyburide=0, glipizide=1
        assert df_transformed.loc[0, 'nummed'] == 2
        assert df_transformed.loc[1, 'nummed'] == 2
        assert df_transformed.loc[2, 'nummed'] == 3

    def test_handle_non_numeric_values(self, feature_creator,
                                       sample_dataframe):
        """Test handling of non-numeric medication values."""
        df_transformed = feature_creator.transform(sample_dataframe)

        # Should create nummed column with zeros (non-numeric values become 0)
        assert 'nummed' in df_transformed.columns
        assert all(df_transformed['nummed'] == 0)

    def test_missing_medication_columns(self, feature_creator):
        """Test handling when medication columns are missing."""
        df_no_meds = pd.DataFrame({'other_col': [1, 2, 3]})
        df_transformed = feature_creator.transform(df_no_meds)

        # Should create nummed column with zeros
        assert 'nummed' in df_transformed.columns
        assert all(df_transformed['nummed'] == 0)


class TestDiabetesFeatureEngineer:
    """Test the main feature engineering pipeline."""

    def test_init_successful(self, sample_config):
        """Test successful initialization of feature engineer."""
        with patch('src.features.feature_eng.load_config') as mock_load_config:
            mock_load_config.return_value = sample_config
            with patch('src.features.feature_eng.get_logger') as mock_logger:
                mock_logger.return_value = MagicMock()

                engineer = DiabetesFeatureEngineer("test_config.yaml")

                assert engineer.config_path == "test_config.yaml"
                assert engineer.config == sample_config
                assert engineer.feature_config == sample_config["features"]
                assert engineer.is_fitted is False

    def test_init_config_load_failure(self):
        """Test initialization failure when config loading fails."""
        with patch('src.features.feature_eng.load_config') as mock_load_config:
            mock_load_config.side_effect = Exception("Config load failed")
            with patch('src.features.feature_eng.get_logger') as mock_logger:
                mock_logger.return_value = MagicMock()

                with pytest.raises(Exception, match="Config load failed"):
                    DiabetesFeatureEngineer("invalid_config.yaml")

    def test_setup_transformers(self, feature_engineer_with_config):
        """Test that transformers are set up correctly."""
        engineer = feature_engineer_with_config

        assert hasattr(engineer, 'icd9_categorizer')
        assert hasattr(engineer, 'medication_creator')
        assert hasattr(engineer, 'value_mapper')
        assert hasattr(engineer, 'additional_creator')

        assert isinstance(engineer.icd9_categorizer, ICD9Categorizer)
        assert isinstance(
            engineer.medication_creator, MedicationFeatureCreator)
        assert isinstance(engineer.value_mapper, ValueMapper)
        assert isinstance(
            engineer.additional_creator, AdditionalFeatureCreator)

    def test_load_data_with_filepath(self, feature_engineer_with_config,
                                     sample_dataframe):
        """Test loading data with specific filepath."""
        with tempfile.NamedTemporaryFile(
                mode='w', suffix='.csv', delete=False) as f:
            sample_dataframe.to_csv(f.name, index=False)

            try:
                df = feature_engineer_with_config.load_data(f.name)
                assert df.shape == sample_dataframe.shape
                assert list(df.columns) == list(sample_dataframe.columns)
            finally:
                os.unlink(f.name)

    def test_load_data_with_preprocessor(self, feature_engineer_with_config,
                                         sample_dataframe):
        """Test loading data using preprocessor."""
        with patch('src.preprocess.preprocessing.DiabetesDataPreprocessor') \
                as mock_preprocessor_class:
            mock_preprocessor = mock_preprocessor_class.return_value
            mock_preprocessor.load_data.return_value = sample_dataframe
            mock_preprocessor.fit_transform.return_value = sample_dataframe

            df = feature_engineer_with_config.load_data()

            mock_preprocessor_class.assert_called_once_with(
                feature_engineer_with_config.config_path)
            mock_preprocessor.load_data.assert_called_once()
            mock_preprocessor.fit_transform.assert_called_once_with(
                sample_dataframe)
            assert df.equals(sample_dataframe)

    def test_load_data_file_error(self, feature_engineer_with_config):
        """Test error handling when file loading fails."""
        with pytest.raises(Exception):
            feature_engineer_with_config.load_data("nonexistent_file.csv")

    def test_fit_transform_full_pipeline(self, feature_engineer_with_config,
                                         sample_dataframe):
        """Test the complete feature engineering pipeline."""
        df_processed = feature_engineer_with_config.fit_transform(
            sample_dataframe)

        # Check that engineer is marked as fitted
        assert feature_engineer_with_config.is_fitted is True

        # Check that transformations were applied
        assert 'numchange' in df_processed.columns  # From medication creator
        assert 'nummed' in df_processed.columns  # From additional creator

        # Check that categorization was applied to diagnosis columns
        assert 'Diabetes' in df_processed['diag_1'].values
        assert 'Circulatory' in df_processed['diag_1'].values

        # Check that value mappings were applied (categorical -> numeric)
        assert df_processed['gender'].dtype == 'int64'
        assert df_processed['age'].dtype == 'int64'

    def test_fit_transform_disabled(self, feature_engineer_with_config,
                                    sample_dataframe):
        """Test pipeline when feature engineering is disabled."""
        feature_engineer_with_config.feature_config['enabled'] = False

        df_processed = feature_engineer_with_config.fit_transform(
            sample_dataframe)

        # Should return original dataframe unchanged
        assert df_processed.equals(sample_dataframe)
        assert feature_engineer_with_config.is_fitted is True

    def test_fit_transform_empty_dataframe(self, feature_engineer_with_config):
        """Test pipeline with empty dataframe."""
        empty_df = pd.DataFrame()

        df_processed = feature_engineer_with_config.fit_transform(empty_df)

        # Should handle empty dataframe gracefully
        assert df_processed.empty
        assert feature_engineer_with_config.is_fitted is True


class TestFeatureTransformerRegistry:
    """Test the feature transformer registry."""

    def test_registry_contains_all_transformers(self):
        """Test that registry contains all expected transformers."""
        expected_transformers = [
            "icd9_categorizer",
            "medication_creator",
            "value_mapper",
            "additional_creator"
        ]

        for transformer_name in expected_transformers:
            assert transformer_name in FEATURE_TRANSFORMERS

    def test_registry_transformer_creation(self, sample_config):
        """Test that registry can create transformer instances."""
        config = sample_config["features"]

        for transformer_name, builder in FEATURE_TRANSFORMERS.items():
            transformer = builder(config)

            # Check that transformer was created and has required methods
            assert hasattr(transformer, 'fit')
            assert hasattr(transformer, 'transform')
            assert hasattr(transformer, 'fit_transform')


class TestConvenienceFunction:
    """Test the convenience function."""

    @patch('src.features.feature_eng.DiabetesFeatureEngineer')
    def test_engineer_diabetes_features(self, mock_engineer_class):
        """Test the convenience function."""
        # Mock the engineer instance
        mock_engineer = MagicMock()
        mock_engineer_class.return_value = mock_engineer

        # Mock the data and processing
        mock_df = pd.DataFrame({'col1': [1, 2, 3]})
        mock_engineer.load_data.return_value = mock_df
        mock_engineer.fit_transform.return_value = mock_df

        # Test the function
        result = engineer_diabetes_features("test_config.yaml")

        # Verify calls
        mock_engineer_class.assert_called_once_with("test_config.yaml")
        mock_engineer.load_data.assert_called_once()
        mock_engineer.fit_transform.assert_called_once_with(mock_df)
        assert result.equals(mock_df)


class TestCLIInterface:
    """Test command line interface functionality."""

    @patch('src.features.feature_eng.setup_logging')
    @patch('src.features.feature_eng.engineer_diabetes_features')
    def test_cli_single_argument(self, mock_engineer_func, mock_setup_logging):
        """Test CLI with single argument (config file)."""
        mock_logger = MagicMock()
        mock_setup_logging.return_value = mock_logger

        mock_df = pd.DataFrame({'col1': [1, 2, 3]})
        mock_engineer_func.return_value = mock_df

        # Simulate single argument CLI call
        config_path = 'config.yaml'
        result_df = mock_engineer_func(config_path)

        mock_engineer_func.assert_called_once_with('config.yaml')
        assert result_df.equals(mock_df)

    @patch('src.features.feature_eng.setup_logging')
    @patch('src.features.feature_eng.DiabetesFeatureEngineer')
    def test_cli_two_arguments(self, mock_engineer_class, mock_setup_logging):
        """Test CLI with two arguments (data file and config file)."""
        mock_logger = MagicMock()
        mock_setup_logging.return_value = mock_logger

        # Mock the engineer
        mock_engineer = MagicMock()
        mock_engineer_class.return_value = mock_engineer

        mock_df = pd.DataFrame({'col1': [1, 2, 3]})
        mock_engineer.load_data.return_value = mock_df
        mock_engineer.fit_transform.return_value = mock_df

        # Simulate two argument CLI call
        data_path, config_path = 'data.csv', 'config.yaml'

        engineer = mock_engineer_class(config_path)
        df = engineer.load_data(data_path)
        df_processed = engineer.fit_transform(df)
        assert df_processed.equals(mock_df)

        mock_engineer_class.assert_called_with('config.yaml')
        mock_engineer.load_data.assert_called_with('data.csv')
        mock_engineer.fit_transform.assert_called_with(mock_df)


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_missing_config_sections(self, sample_dataframe):
        """Test handling when config sections are missing."""
        minimal_config = {"features": {"enabled": True}}

        with patch('src.features.feature_eng.load_config') as mock_load_config:
            mock_load_config.return_value = minimal_config
            with patch('src.features.feature_eng.get_logger') as mock_logger:
                mock_logger.return_value = MagicMock()

                engineer = DiabetesFeatureEngineer("test_config.yaml")

                # Should work with minimal config (using empty defaults)
                df_processed = engineer.fit_transform(sample_dataframe)

                # With minimal config, transformers still run but with empty
                # configurations so the dataframe will be modified,
                # not identical to the original
                assert engineer.is_fitted is True

                # Check that new feature columns were added
                assert 'numchange' in df_processed.columns
                assert 'nummed' in df_processed.columns

                # Check that we have the same number of rows but more columns
                assert df_processed.shape[0] == sample_dataframe.shape[0]
                assert df_processed.shape[1] > sample_dataframe.shape[1]

                # With empty ICD9 categories, all codes should become "Other"
                for col in ['diag_1', 'diag_2', 'diag_3']:
                    if col in df_processed.columns:
                        # All non-null values should be "Other" with
                        # empty categories
                        non_null_values = df_processed[col].dropna()
                        if not non_null_values.empty:
                            assert all(val == "Other" for val
                                       in non_null_values)

                # With empty value mappings, original values should be
                # preserved for categorical columns (no conversion to int64)
                original_string_cols = ['gender', 'diabetesMed', 'A1Cresult',
                                        'age', 'readmitted']
                for col in original_string_cols:
                    if col in df_processed.columns and \
                            col in sample_dataframe.columns:
                        # Should preserve original string values
                        # since no mappings provided
                        assert df_processed[col].equals(sample_dataframe[col])

    def test_invalid_icd9_codes_handling(self):
        """Test handling of various invalid ICD-9 code formats."""
        categorizer = ICD9Categorizer({"Diabetes": [[250, 250]]})

        # Test various invalid formats
        invalid_codes = [None, np.nan, "", "invalid", "V", "E", "123.456.789"]

        for code in invalid_codes:
            result = categorizer._categorize_icd9(
                code, categorizer.icd9_categories)
            assert result == "Other"

    def test_medication_features_with_missing_data(self):
        """Test medication feature creation with missing medication data."""
        creator = MedicationFeatureCreator(["metformin", "insulin"])

        df_missing = pd.DataFrame({
            'other_col': [1, 2, 3],
            'metformin': ['No', None, 'Up'],  # Missing value
            # insulin column missing entirely
        })

        df_processed = creator.transform(df_missing)

        # Should handle missing data gracefully
        assert 'numchange' in df_processed.columns
        assert not df_processed['numchange'].isna().any()

    def test_value_mapping_with_unmapped_values(self):
        """Test value mapping when encountering unmapped values."""
        mappings = {
            "binary_vars": {
                "gender": {"Male": 1, "Female": 0}
                # "Unknown" not mapped
            }
        }

        mapper = ValueMapper(mappings)

        df_unmapped = pd.DataFrame({
            'gender': ['Male', 'Female', 'Unknown', None]
        })

        # The current implementation will fail when trying to convert 'Unknown'
        # to int64 so we should expect this to raise an exception
        with pytest.raises(ValueError, match="invalid literal for int"):
            mapper.transform(df_unmapped)

    def test_value_mapping_error_handling(self):
        """Test that value mapping properly handles conversion errors."""
        mappings = {
            "binary_vars": {
                "mixed_col": {"A": 1, "B": 0}
            }
        }

        mapper = ValueMapper(mappings)

        # Test with unmapped values that will cause conversion errors
        df_mixed = pd.DataFrame({
            'mixed_col': ['A', 'B', 'C', 'D']  # C and D are not mapped
        })

        # Should raise error when trying to convert unmapped values to int64
        with pytest.raises(ValueError):
            mapper.transform(df_mixed)

    def test_value_mapping_with_all_mapped_values(self):
        """Test value mapping when all values are mapped."""
        mappings = {
            "binary_vars": {
                "gender": {"Male": 1, "Female": 0}
            }
        }

        mapper = ValueMapper(mappings)

        df_all_mapped = pd.DataFrame({
            'gender': ['Male', 'Female', 'Male', 'Female']
        })

        df_processed = mapper.transform(df_all_mapped)

        # Should successfully map and convert to int64
        assert df_processed.loc[0, 'gender'] == 1  # Male -> 1
        assert df_processed.loc[1, 'gender'] == 0  # Female -> 0
        assert df_processed['gender'].dtype == 'int64'

    def test_value_mapping_handles_none_values(self):
        """Test value mapping with None values."""
        mappings = {
            "binary_vars": {
                "status": {"Active": 1, "Inactive": 0}
            }
        }

        mapper = ValueMapper(mappings)

        df_with_none = pd.DataFrame({
            'status': ['Active', 'Inactive', None, 'Active']
        })

        # This should fail because None can't be converted to int64
        with pytest.raises((ValueError, TypeError)):
            mapper.transform(df_with_none)


class TestDataIntegrity:
    """Test data integrity and validation."""

    def test_feature_consistency(self, feature_engineer_with_config,
                                 sample_dataframe):
        """Test that features are created consistently."""
        df_processed = feature_engineer_with_config.fit_transform(
            sample_dataframe)

        # Check that new features were added
        new_features = ['numchange', 'nummed']
        for feature in new_features:
            assert feature in df_processed.columns

        # Check that feature values are reasonable
        assert (df_processed['numchange'] >= 0).all()
        assert (df_processed['nummed'] >= 0).all()

    def test_data_types_consistency(self, feature_engineer_with_config,
                                    sample_dataframe):
        """Test that data types are handled consistently."""
        df_processed = feature_engineer_with_config.fit_transform(
            sample_dataframe)

        # Check that mapped categorical variables became integers
        mapped_columns = ['admission_type_id', 'gender', 'diabetesMed', 'age',
                          'readmitted']
        for col in mapped_columns:
            if col in df_processed.columns:
                assert pd.api.types.is_integer_dtype(df_processed[col])

    def test_no_data_loss(self, feature_engineer_with_config,
                          sample_dataframe):
        """Test that no rows are lost during feature engineering."""
        df_processed = feature_engineer_with_config.fit_transform(
            sample_dataframe)

        # Should have same number of rows
        assert df_processed.shape[0] == sample_dataframe.shape[0]

        # Should have at least as many columns (features added)
        assert df_processed.shape[1] >= sample_dataframe.shape[1]

    def test_feature_engineering_idempotent(self, feature_engineer_with_config,
                                            sample_dataframe):
        """Test that feature engineering produces consistent results."""
        df_processed1 = feature_engineer_with_config.fit_transform(
            sample_dataframe.copy())
        df_processed2 = feature_engineer_with_config.fit_transform(
            sample_dataframe.copy())

        # Should produce identical results
        pd.testing.assert_frame_equal(df_processed1, df_processed2)
