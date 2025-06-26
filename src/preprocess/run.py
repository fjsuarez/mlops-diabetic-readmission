"""
preprocess/run.py

MLflow-compatible, modular preprocessing step with Hydra config, W&B logging,
and robust error handling. Handles data cleaning, feature engineering, and
preparation for downstream ML tasks.
"""

import sys
import logging
import os
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import json
import pandas as pd
import tempfile
import hashlib
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from preprocess.preprocessing import DiabetesDataPreprocessor  # noqa: E402

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("preprocess")


def compute_df_hash(df: pd.DataFrame) -> str:
    """Compute a hash for the input DataFrame for traceability."""
    return hashlib.sha256(pd.util.hash_pandas_object(
        df, index=True).values).hexdigest()


def _create_preprocessing_summary(df_input: pd.DataFrame,
                                  df_output: pd.DataFrame) -> Dict[str, Any]:
    """Create a summary of preprocessing transformations."""
    return {
        "input_shape": df_input.shape,
        "output_shape": df_output.shape,
        "rows_removed": df_input.shape[0] - df_output.shape[0],
        "columns_removed": df_input.shape[1] - df_output.shape[1],
        "input_columns": list(df_input.columns),
        "output_columns": list(df_output.columns),
        "columns_added": [col for col in df_output.columns
                          if col not in df_input.columns],
        "columns_dropped": [col for col in df_input.columns
                            if col not in df_output.columns]
    }


@hydra.main(
    config_path=str(PROJECT_ROOT),
    config_name="config",
    version_base=None)
def main(cfg: DictConfig) -> None:
    config_path = PROJECT_ROOT / "config.yaml"

    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"preprocess_{dt_str}"

    run = None
    try:
        run = wandb.init(
            project=cfg.main.WANDB_PROJECT,
            entity=cfg.main.WANDB_ENTITY,
            job_type="preprocess",
            name=run_name,
            config=dict(cfg),
            tags=["preprocess"]
        )
        logger.info("Started WandB run: %s", run_name)

        # Load validated data artifact from W&B
        val_art = run.use_artifact("validated_data:latest")
        with tempfile.TemporaryDirectory() as tmp_dir:
            val_path = val_art.download(root=tmp_dir)
            # Use the same CSV reading parameters as data_validation
            data_source_config = OmegaConf.to_container(cfg.data_source,
                                                        resolve=True)
            csv_params = {
                "keep_default_na": data_source_config.get("keep_default_na",
                                                          True),
                "na_values": data_source_config.get("na_values", None)
            }

            # Read CSV with config parameters (consistent with validation step)
            df = pd.read_csv(
                os.path.join(val_path, "validated_data.csv"),
                **csv_params
            )

        if df.empty:
            logger.warning("Loaded dataframe is empty.")

        # Log input data hash for traceability
        input_hash = compute_df_hash(df)
        wandb.summary.update({"input_data_hash": input_hash})

        # Log input schema and sample
        input_schema = {col: str(dtype) for col, dtype in df.dtypes.items()}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json',
                                         delete=False) as f:
            json.dump(input_schema, f, indent=2)
            input_schema_path = f.name

        # Log input sample artifacts
        if cfg.data_load.get("log_sample_artifacts", True):
            sample_tbl = wandb.Table(dataframe=df.head(100))
            wandb.log({"input_sample_rows": sample_tbl})

        if cfg.data_load.get("log_summary_stats", True):
            stats_tbl = wandb.Table(
                dataframe=df.describe(include="all").T.reset_index()
            )
            wandb.log({"input_summary_stats": stats_tbl})

        # Initialize preprocessor and run preprocessing
        preprocessor = DiabetesDataPreprocessor(str(config_path))
        df_processed = preprocessor.fit_transform(df)

        # Log output data hash
        output_hash = compute_df_hash(df_processed)
        wandb.summary.update({"output_data_hash": output_hash})

        # Create preprocessing summary
        preprocessing_summary = _create_preprocessing_summary(df, df_processed)
        wandb.summary.update(preprocessing_summary)

        # Log output schema
        output_schema = {col: str(dtype) for col, dtype
                         in df_processed.dtypes.items()}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json',
                                         delete=False) as f:
            json.dump(output_schema, f, indent=2)
            output_schema_path = f.name

        # Log output sample artifacts
        if cfg.data_load.get("log_sample_artifacts", True):
            output_sample_tbl = wandb.Table(dataframe=df_processed.head(100))
            wandb.log({"output_sample_rows": output_sample_tbl})

        if cfg.data_load.get("log_summary_stats", True):
            output_stats_tbl = wandb.Table(
                dataframe=df_processed.describe(include="all").T.reset_index()
            )
            wandb.log({"output_summary_stats": output_stats_tbl})

        # Save processed data to temporary file and log to W&B
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "preprocessed_data.csv"
            df_processed.to_csv(tmp_path, index=False)

            if cfg.data_load.get("log_artifacts", True):
                preprocess_artifact = wandb.Artifact("preprocessed_data",
                                                     type="dataset")
                preprocess_artifact.add_file(str(tmp_path),
                                             name="preprocessed_data.csv")
                preprocess_artifact.add_file(input_schema_path,
                                             name="input_schema.json")
                preprocess_artifact.add_file(output_schema_path,
                                             name="output_schema.json")
                run.log_artifact(preprocess_artifact, aliases=["latest"])
                logger.info("Logged preprocessed data artifact to WandB")

        # Log preprocessing transformation details
        preprocessing_config = OmegaConf.to_container(cfg.preprocessing,
                                                      resolve=True)
        transform_details = {
            "preprocessing_enabled": preprocessing_config.get("enabled", True),
            "drop_columns": preprocessing_config.get("drop_columns", []),
            "exclude_discharge_disposition": preprocessing_config.get(
                "exclude_discharge_disposition", []),
            "na_indicators": preprocessing_config.get("na_indicators", []),
            "duplicate_subset": preprocessing_config.get(
                "duplicate_subset", ["patient_nbr"])
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json',
                                         delete=False) as f:
            json.dump(transform_details, f, indent=2)
            transform_details_path = f.name

        # Create preprocessing report artifact
        if cfg.data_load.get("log_artifacts", True):
            report_artifact = wandb.Artifact("preprocessing_report",
                                             type="report")
            report_artifact.add_file(transform_details_path,
                                     name="preprocessing_config.json")
            run.log_artifact(report_artifact, aliases=["latest"])
            logger.info("Logged preprocessing report to WandB")

        # Log final summary metrics
        wandb.summary.update({
            "preprocessing_result": "success",
            "input_rows": df.shape[0],
            "output_rows": df_processed.shape[0],
            "input_columns": df.shape[1],
            "output_columns": df_processed.shape[1],
            "data_reduction_percent": round(
                (1 - df_processed.shape[0] /
                 df.shape[0]) * 100, 2) if df.shape[0] > 0 else 0})

        logger.info("Preprocessing completed successfully")
        logger.info(f"Input shape: {df.shape}")
        logger.info(f"Output shape: {df_processed.shape}")
        logger.info(f"Rows removed: {df.shape[0] - df_processed.shape[0]}")

        # Clean up temporary files
        os.unlink(input_schema_path)
        os.unlink(output_schema_path)
        os.unlink(transform_details_path)

    except Exception as e:
        logger.exception("Failed during preprocessing step")
        if run is not None:
            run.alert(title="Preprocessing Error", text=str(e))
            wandb.summary.update({"preprocessing_result": "failed"})
        sys.exit(1)
    finally:
        if wandb.run is not None:
            wandb.finish()
            logger.info("WandB run finished")


if __name__ == "__main__":
    main()
