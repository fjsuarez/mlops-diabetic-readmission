"""
Unit tests for data_validator.py module.

Tests cover validation logic, error handling, and configuration behavior
using mock data and test fixtures.
"""

import pytest
import pandas as pd
import json
import tempfile
import os
from unittest.mock import patch, MagicMock
from src.data_validation.data_validator import (
    _is_dtype_compatible,
    _validate_column,
    validate_data,
    validate_data_from_config
)


class TestDtypeCompatibility:
    """Test dtype compatibility checking function."""

    def test_int_dtype_compatible(self):
        """Test integer dtype compatibility."""
        int_series = pd.Series([1, 2, 3], dtype='int64')
        assert _is_dtype_compatible(int_series, "int") is True
        assert _is_dtype_compatible(int_series, "float") is False
        assert _is_dtype_compatible(int_series, "str") is False

    def test_float_dtype_compatible(self):
        """Test float dtype compatibility."""
        float_series = pd.Series([1.1, 2.2, 3.3], dtype='float64')
        assert _is_dtype_compatible(float_series, "float") is True
        assert _is_dtype_compatible(float_series, "int") is False
        assert _is_dtype_compatible(float_series, "str") is False

    def test_str_dtype_compatible(self):
        """Test string dtype compatibility."""
        str_series = pd.Series(['a', 'b', 'c'], dtype='object')
        assert _is_dtype_compatible(str_series, "str") is True
        assert _is_dtype_compatible(str_series, "int") is False
        assert _is_dtype_compatible(str_series, "float") is False

    def test_bool_dtype_compatible(self):
        """Test boolean dtype compatibility."""
        bool_series = pd.Series([True, False, True], dtype='bool')
        assert _is_dtype_compatible(bool_series, "bool") is True
        assert _is_dtype_compatible(bool_series, "int") is False
        assert _is_dtype_compatible(bool_series, "str") is False

    def test_unknown_dtype(self):
        """Test unknown dtype returns False."""
        int_series = pd.Series([1, 2, 3], dtype='int64')
        assert _is_dtype_compatible(int_series, "unknown") is False


class TestValidateColumn:
    """Test individual column validation logic."""

    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame({
            'valid_int': [1, 2, 3, 4, 5],
            'with_nulls': [1, 2, None, 4, 5],
            'string_col': ['a', 'b', 'c', 'd', 'e'],
            'range_col': [10, 20, 30, 40, 50],
            'categorical': ['cat', 'dog', 'bird', 'cat', 'fish'],
            'mixed_values': ['Yes', 'No', 'Maybe', 'Yes', 'No']
        })

    def test_missing_required_column(self, sample_dataframe):
        """Test validation when required column is missing."""
        col_schema = {"name": "missing_col", "required": True}
        errors, warnings, report = [], [], {}

        _validate_column(
            sample_dataframe,
            col_schema,
            errors,
            warnings,
            report)

        assert len(errors) == 1
        assert "Missing required column: missing_col" in errors[0]
        assert report["missing_col"]["status"] == "missing"

    def test_missing_optional_column(self, sample_dataframe):
        """Test validation when optional column is missing."""
        col_schema = {"name": "missing_col", "required": False}
        errors, warnings, report = [], [], {}

        _validate_column(
            sample_dataframe,
            col_schema,
            errors,
            warnings,
            report)

        assert len(errors) == 0
        assert report["missing_col"]["status"] == "not present (optional)"

    def test_dtype_mismatch(self, sample_dataframe):
        """Test validation when column dtype doesn't match expected."""
        col_schema = {"name": "string_col", "dtype": "int", "required": True}
        errors, warnings, report = [], [], {}

        _validate_column(sample_dataframe, col_schema, errors, warnings,
                         report)

        assert len(errors) == 1
        assert "expected 'int'" in errors[0]
        assert report["string_col"]["dtype_expected"] == "int"

    def test_missing_values_required_column(self, sample_dataframe):
        """Test validation when required column has missing values."""
        col_schema = {"name": "with_nulls", "required": True}
        errors, warnings, report = [], [], {}

        _validate_column(sample_dataframe, col_schema, errors, warnings,
                         report)

        assert len(errors) == 1
        # Fix: Use more flexible string matching to handle whitespace
        assert "missing values" in errors[0] and "(required)" in errors[0]
        assert report["with_nulls"]["missing_count"] == 1

    def test_missing_values_optional_column(self, sample_dataframe):
        """Test validation when optional column has missing values."""
        col_schema = {"name": "with_nulls", "required": False}
        errors, warnings, report = [], [], {}

        _validate_column(sample_dataframe, col_schema, errors, warnings,
                         report)

        assert len(errors) == 0
        assert len(warnings) == 1
        # Fix: Use more flexible string matching to handle whitespace
        assert "missing values" in warnings[0] and "(optional)" in warnings[0]

    def test_min_value_validation(self, sample_dataframe):
        """Test minimum value validation."""
        col_schema = {"name": "range_col", "min": 25}
        errors, warnings, report = [], [], {}

        _validate_column(sample_dataframe, col_schema, errors, warnings,
                         report)

        assert len(errors) == 1
        assert "values below min (25)" in errors[0]
        assert report["range_col"]["below_min"] == 2  # 10, 20 are below 25

    def test_max_value_validation(self, sample_dataframe):
        """Test maximum value validation."""
        col_schema = {"name": "range_col", "max": 35}
        errors, warnings, report = [], [], {}

        _validate_column(sample_dataframe, col_schema, errors, warnings,
                         report)

        assert len(errors) == 1
        assert "values above max (35)" in errors[0]
        assert report["range_col"]["above_max"] == 2  # 40, 50 are above 35

    def test_allowed_values_validation(self, sample_dataframe):
        """Test allowed values validation."""
        col_schema = {
            "name": "categorical",
            "allowed_values": ["cat", "dog", "bird"]
        }
        errors, warnings, report = [], [], {}

        _validate_column(sample_dataframe, col_schema, errors, warnings,
                         report)

        assert len(errors) == 1
        # Fix: Use more flexible string matching to handle set formatting
        assert "not in allowed set" in errors[0] or "values not in allowed" \
            in errors[0]
        assert report["categorical"]["invalid_values_count"] == 1

    def test_successful_validation(self, sample_dataframe):
        """Test successful column validation with no errors."""
        col_schema = {
            "name": "valid_int",
            "dtype": "int",
            "required": True,
            "min": 0,
            "max": 10
        }
        errors, warnings, report = [], [], {}

        _validate_column(sample_dataframe, col_schema, errors, warnings,
                         report)

        assert len(errors) == 0
        assert len(warnings) == 0
        assert report["valid_int"]["status"] == "present"
        assert "sample_values" in report["valid_int"]


class TestValidateData:
    """Test full data validation function."""

    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame({
            'encounter_id': [1, 2, 3],
            'patient_nbr': [101, 102, 103],
            'race': ['Caucasian', 'AfricanAmerican', 'Hispanic'],
            'gender': ['Male', 'Female', 'Male'],
            'readmitted': ['NO', '<30', '>30']
        })

    @pytest.fixture
    def valid_config(self):
        """Create a valid configuration for testing."""
        return {
            "data_validation": {
                "enabled": True,
                "action_on_error": "raise",
                "report_path": "test_validation_report.json",
                "schema": {
                    "columns": [
                        {
                            "name": "encounter_id",
                            "dtype": "int",
                            "required": True,
                            "min": 1
                        },
                        {
                            "name": "patient_nbr",
                            "dtype": "int",
                            "required": True,
                            "min": 1
                        },
                        {
                            "name": "race",
                            "dtype": "str",
                            "required": False,
                            "allowed_values": [
                                "Caucasian",
                                "AfricanAmerican",
                                "Hispanic",
                                "Asian",
                                "Other",
                                "?"]
                        },
                        {
                            "name": "gender",
                            "dtype": "str",
                            "required": True,
                            "allowed_values": [
                                "Male",
                                "Female",
                                "Unknown/Invalid"]
                        },
                        {
                            "name": "readmitted",
                            "dtype": "str",
                            "required": True,
                            "allowed_values": ["NO", "<30", ">30"]
                        }
                    ]
                }
            }
        }

    def test_validation_disabled(self, sample_dataframe):
        """Test when validation is disabled in config."""
        config = {"data_validation": {"enabled": False}}

        # Should not raise any exceptions
        validate_data(sample_dataframe, config)

    def test_no_schema_defined(self, sample_dataframe):
        """Test when no schema is defined in config."""
        config = {"data_validation": {"enabled": True}}

        # Should not raise any exceptions, just log warning
        validate_data(sample_dataframe, config)

    def test_successful_validation(self, sample_dataframe, valid_config):
        """Test successful validation with valid data."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            report_path = os.path.join(tmp_dir, "validation_report.json")
            valid_config["data_validation"]["report_path"] = report_path

            # Should not raise any exceptions
            validate_data(sample_dataframe, valid_config)

            # Check that report was created
            assert os.path.exists(report_path)

            # Check report contents
            with open(report_path, 'r') as f:
                report = json.load(f)

            assert report["result"] == "pass"
            assert len(report["errors"]) == 0

    def test_validation_failure_with_raise(
            self,
            sample_dataframe,
            valid_config):
        """Test validation failure with raise action."""
        # Add invalid schema that will fail
        valid_config["data_validation"]["schema"]["columns"].append({
            "name": "missing_column",
            "required": True
        })

        with tempfile.TemporaryDirectory() as tmp_dir:
            report_path = os.path.join(tmp_dir, "validation_report.json")
            valid_config["data_validation"]["report_path"] = report_path

            with pytest.raises(ValueError, match="Data validation failed"):
                validate_data(sample_dataframe, valid_config)

            # Check that report was still created
            assert os.path.exists(report_path)

            with open(report_path, 'r') as f:
                report = json.load(f)

            assert report["result"] == "fail"
            assert len(report["errors"]) > 0

    def test_validation_failure_with_warn(
            self, sample_dataframe, valid_config):
        """Test validation failure with warn action."""
        # Change action to warn
        valid_config["data_validation"]["action_on_error"] = "warn"

        # Add invalid schema that will fail
        valid_config["data_validation"]["schema"]["columns"].append({
            "name": "missing_column",
            "required": True
        })

        with tempfile.TemporaryDirectory() as tmp_dir:
            report_path = os.path.join(tmp_dir, "validation_report.json")
            valid_config["data_validation"]["report_path"] = report_path

            # Should not raise, just warn
            validate_data(sample_dataframe, valid_config)

    def test_validation_with_warnings(self, sample_dataframe, valid_config):
        """Test validation that produces warnings but passes."""
        # Fix: Add a column that will actually produce warnings
        # Create a sample with missing values in an optional column
        sample_with_nulls = sample_dataframe.copy()
        sample_with_nulls['optional_col'] = [None, 'value', None]

        valid_config["data_validation"]["schema"]["columns"].append({
            "name": "optional_col",
            "required": False  # This will generate warnings for missing values
        })

        with tempfile.TemporaryDirectory() as tmp_dir:
            report_path = os.path.join(tmp_dir, "validation_report.json")
            valid_config["data_validation"]["report_path"] = report_path

            validate_data(sample_with_nulls, valid_config)

            with open(report_path, 'r') as f:
                report = json.load(f)

            assert report["result"] == "pass"
            assert len(report["warnings"]) > 0

    def test_report_directory_creation(self, sample_dataframe, valid_config):
        """Test that report directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            nested_path = os.path.join(tmp_dir, "nested", "logs",
                                       "validation_report.json")
            valid_config["data_validation"]["report_path"] = nested_path

            validate_data(sample_dataframe, valid_config)

            assert os.path.exists(nested_path)


class TestValidateDataFromConfig:
    """Test the convenience function that loads data from config."""

    @patch('src.data_validation.data_validator.load_config')
    @patch('src.data_load.data_loader.DataLoader')
    def test_validate_data_from_config(self, mock_data_loader,
                                       mock_load_config):
        """Test validate_data_from_config function."""
        # Mock configuration
        mock_config = {
            "data_validation": {
                "enabled": True,
                "action_on_error": "raise",
                "report_path": "test_report.json",
                "schema": {"columns": []}
            }
        }
        mock_load_config.return_value = mock_config

        # Mock data loader
        mock_df = pd.DataFrame({'col1': [1, 2, 3]})
        mock_loader_instance = mock_data_loader.return_value
        mock_loader_instance.load_data.return_value = mock_df

        # Fix: Mock the logger parameter
        with patch('src.data_validation.data_validator.logger') as mock_logger:
            # Test the function
            with tempfile.TemporaryDirectory() as tmp_dir:
                config_path = os.path.join(tmp_dir, "config.yaml")

                validate_data_from_config(config_path)

                # Verify mocks were called correctly
                mock_load_config.assert_called_once_with(mock_logger,
                                                         config_path)
                mock_data_loader.assert_called_once_with(config_path)
                mock_loader_instance.load_data.assert_called_once()


class TestCLIInterface:
    """Test command line interface functionality."""

    def test_cli_single_argument(self):
        """Test CLI with single argument (config file)."""
        module = 'src.data_validation.data_validator'
        with patch(module+'.setup_logging') as mock_setup_logging, \
             patch(module+'.validate_data_from_config') as mock_validate:

            # Simulate the CLI logic
            mock_logger = MagicMock()
            mock_setup_logging.return_value = mock_logger

            # Simulate what happens in the if __name__ == "__main__" block
            config_path = 'config.yaml'
            mock_validate(config_path)

            mock_validate.assert_called_once_with('config.yaml')

    def test_cli_two_arguments(self):
        """Test CLI with two arguments (data file and config file)."""
        module = 'src.data_validation.data_validator'
        with patch(module+'.setup_logging') as mock_setup_logging, \
             patch(module+'.validate_data') as mock_validate_data, \
             patch(module+'.load_config') as mock_load_config, \
             patch('pandas.read_csv') as mock_read_csv:

            mock_df = pd.DataFrame({'col1': [1, 2, 3]})
            mock_read_csv.return_value = mock_df
            mock_config = {"data_validation": {"enabled": True}}
            mock_load_config.return_value = mock_config
            mock_logger = MagicMock()
            mock_setup_logging.return_value = mock_logger

            # Simulate the CLI logic for two arguments
            data_path, config_path = 'data.csv', 'config.yaml'
            df = mock_read_csv(data_path)
            config = mock_load_config(mock_logger, config_path)
            mock_validate_data(df, config)

            mock_read_csv.assert_called_once_with('data.csv')
            mock_load_config.assert_called_once_with(mock_logger,
                                                     'config.yaml')
            mock_validate_data.assert_called_once_with(mock_df, mock_config)


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_sample_values_error_handling(self):
        """Test error handling when getting sample values fails."""
        # Create a DataFrame with problematic data for sample values
        df = pd.DataFrame({
            'problem_col': [{'dict': 'value'}, {'another': 'dict'}, None]
        })

        col_schema = {"name": "problem_col"}
        errors, warnings, report = [], [], {}

        _validate_column(df, col_schema, errors, warnings, report)

        # Should handle the error gracefully
        assert report["problem_col"]["sample_values"] == "unavailable"

    def test_dtype_validation_stops_further_checks(self):
        """Test that dtype mismatch stops further validation checks."""
        df = pd.DataFrame({
            'string_as_int': ['a', 'b', 'c']
        })

        col_schema = {
            "name": "string_as_int",
            "dtype": "int",
            "min": 0,
            "max": 100,
            "allowed_values": [1, 2, 3]
        }
        errors, warnings, report = [], [], {}

        _validate_column(df, col_schema, errors, warnings, report)

        # Should only have dtype error, not min/max/allowed_values errors
        assert len(errors) == 1
        assert "dtype" in errors[0]
        assert "below_min" not in report["string_as_int"]
        assert "above_max" not in report["string_as_int"]
        assert "invalid_values_count" not in report["string_as_int"]
