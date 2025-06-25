import pytest
import pandas as pd
from tempfile import NamedTemporaryFile
import os
from unittest.mock import patch
from src.data_load.data_loader import DataLoader, load_diabetic_data


class TestDataLoader:
    """Test cases for DataLoader class."""

    @pytest.fixture
    def sample_csv_data(self):
        """Create a small mock CSV dataset for testing."""
        return """encounter_id,patient_nbr,race,gender,age,weight,admission_type_id,discharge_disposition_id,admission_source_id,time_in_hospital,payer_code,medical_specialty,num_lab_procedures,num_procedures,num_medications,number_outpatient,number_emergency,number_inpatient,diag_1,diag_2,diag_3,number_diagnoses,max_glu_serum,A1Cresult,metformin,repaglinide,nateglinide,chlorpropamide,glimepiride,acetohexamide,glipizide,glyburide,tolbutamide,pioglitazone,rosiglitazone,acarbose,miglitol,troglitazone,tolazamide,examide,citoglipton,insulin,glyburide-metformin,glipizide-metformin,glimepiride-pioglitazone,metformin-rosiglitazone,metformin-pioglitazone,change,diabetesMed,readmitted
2278392,8222157,Caucasian,Female,[0-10),?,6,25,1,1,?,Pediatrics-Endocrinology,41,0,1,0,0,0,250.83,?,?,1,None,None,No,No,No,No,No,No,No,No,No,No,No,No,No,No,No,No,No,No,No,No,No,No,No,No,No,NO
149190,55629189,Caucasian,Female,[10-20),?,1,1,7,3,?,?,59,0,18,0,0,0,276,250.01,255,9,None,None,No,No,No,No,No,No,No,No,No,No,No,No,No,No,No,No,No,Up,No,No,No,No,No,Ch,Yes,>30
64410,86047875,AfricanAmerican,Female,[20-30),?,1,1,7,2,?,?,11,5,13,2,0,1,648,250,V27,6,None,None,No,No,No,No,No,No,Steady,No,No,No,No,No,No,No,No,No,No,No,No,No,No,No,No,No,Yes,NO"""  # noqa: E501

    @pytest.fixture
    def temp_csv_file(self, sample_csv_data):
        """Create a temporary CSV file for testing."""
        with NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(sample_csv_data)
            temp_file_path = f.name
        yield temp_file_path
        os.unlink(temp_file_path)

    @pytest.fixture
    def temp_config_file(self, temp_csv_file):
        """Create a temporary config file for testing."""
        config_content = f"""data_source:
  raw_path: "{temp_csv_file}"
  type: "csv"
  delimiter: ","
  header: 0
  encoding: "utf-8"
"""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            temp_config_path = f.name
        yield temp_config_path
        os.unlink(temp_config_path)

    def test_data_loader_init_success(self, temp_config_file):
        """Test successful DataLoader initialization."""
        loader = DataLoader(temp_config_file)
        assert loader.config is not None
        assert loader.data_config is not None
        assert "raw_path" in loader.data_config

    def test_data_loader_init_file_not_found(self):
        """Test DataLoader initialization with non-existent config file."""
        with pytest.raises(Exception):
            DataLoader("non_existent_config.yaml")

    def test_load_csv_data_success(self, temp_config_file):
        """Test successful CSV data loading."""
        loader = DataLoader(temp_config_file)
        df = loader.load_data()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3  # Should have 3 rows from sample data
        assert "encounter_id" in df.columns
        assert "patient_nbr" in df.columns
        assert "readmitted" in df.columns

    def test_load_data_file_not_found(self, temp_config_file):
        """Test loading data when file doesn't exist."""
        # Modify config to point to non-existent file
        with open(temp_config_file, "w") as f:
            f.write(
                """data_source:
  raw_path: "non_existent_file.csv"
  type: "csv"
"""
            )

        loader = DataLoader(temp_config_file)
        with pytest.raises(FileNotFoundError):
            loader.load_data()

    def test_load_data_no_raw_path(self):
        """Test loading data when no raw_path is specified."""
        config_content = """data_source:
  type: "csv"
"""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            temp_config_path = f.name

        try:
            loader = DataLoader(temp_config_path)
            with pytest.raises(
                ValueError, match="No file path specified in configuration"
            ):
                loader.load_data()
        finally:
            os.unlink(temp_config_path)

    def test_unsupported_file_type(self, temp_csv_file):
        """Test loading unsupported file type."""
        config_content = f"""data_source:
  raw_path: "{temp_csv_file}"
  type: "unsupported_format"
"""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            temp_config_path = f.name

        try:
            loader = DataLoader(temp_config_path)
            with pytest.raises(ValueError, match="Unsupported file type"):
                loader.load_data()
        finally:
            os.unlink(temp_config_path)

    @patch("src.data_load.data_loader.setup_logging")
    def test_load_diabetic_data_function(
            self,
            mock_setup_logging,
            temp_config_file):
        """Test the convenience function load_diabetic_data."""
        df = load_diabetic_data(config_path=temp_config_file)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        mock_setup_logging.assert_called_once()

    def test_csv_loading_with_custom_delimiter(self, sample_csv_data):
        """Test CSV loading with custom delimiter."""
        # Create CSV with semicolon delimiter
        csv_data_semicolon = sample_csv_data.replace(",", ";")

        with NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_data_semicolon)
            temp_csv_path = f.name

        config_content = f"""data_source:
  raw_path: "{temp_csv_path}"
  type: "csv"
  delimiter: ";"
  header: 0
  encoding: "utf-8"
"""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            temp_config_path = f.name

        try:
            loader = DataLoader(temp_config_path)
            df = loader.load_data()
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 3
        finally:
            os.unlink(temp_csv_path)
            os.unlink(temp_config_path)

    def test_data_shape_and_columns(self, temp_config_file):
        """Test that loaded data has expected shape and columns."""
        loader = DataLoader(temp_config_file)
        df = loader.load_data()

        # Test shape
        assert df.shape[0] == 3  # 3 rows
        assert df.shape[1] == 50  # Expected number of columns

        # Test some key columns exist
        expected_columns = [
            "encounter_id",
            "patient_nbr",
            "race",
            "gender",
            "age",
            "readmitted",
            "diabetesMed",
            "time_in_hospital",
        ]
        for col in expected_columns:
            assert col in df.columns

    @patch("src.data_load.data_loader.pd.read_excel")
    def test_excel_loading(self, mock_read_excel):
        """Test Excel file loading (mocked)."""
        mock_df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        mock_read_excel.return_value = mock_df

        config_content = """data_source:
  raw_path: "test.xlsx"
  type: "excel"
  header: 0
"""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            temp_config_path = f.name

        try:
            loader = DataLoader(temp_config_path)
            # Mock the file existence check
            with patch("os.path.isfile", return_value=True):
                df = loader.load_data()

            assert isinstance(df, pd.DataFrame)
            mock_read_excel.assert_called_once()
        finally:
            os.unlink(temp_config_path)


class TestDataLoaderIntegration:
    """Integration tests using the actual config file."""

    def test_load_with_actual_config(self):
        """Test loading with the actual config.yaml file."""
        # This test uses the real config but might be slow with large data
        # Consider skipping in CI or using a smaller dataset
        try:
            df = load_diabetic_data("config.yaml")
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
            assert "readmitted" in df.columns
        except FileNotFoundError:
            pytest.skip("Actual data file not found - this is expected in CI")
        except Exception as e:
            pytest.fail(f"Unexpected error: {e}")
