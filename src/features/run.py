"""
features/run.py

MLflow-compatible feature engineering step with Hydra config and W&B logging.
Transforms preprocessed data into features ready for model training.
"""

import sys
import os
import tempfile
import json
from datetime import datetime
from pathlib import Path

import hydra
import wandb
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv

# Ensure project modules are importable when executed via MLflow
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.feature_eng import DiabetesFeatureEngineer  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

load_dotenv()
logger = get_logger(__name__)


def _create_feature_summary(
        df_input: pd.DataFrame, df_output: pd.DataFrame) -> dict:
    """Create a summary of feature engineering transformations."""
    return {
        "input_shape": df_input.shape,
        "output_shape": df_output.shape,
        "features_added": df_output.shape[1] - df_input.shape[1],
        "input_columns": list(df_input.columns),
        "output_columns": list(df_output.columns),
        "new_features": [col for col in df_output.columns
                         if col not in df_input.columns],
        "modified_features": [col for col in df_input.columns
                              if col in df_output.columns
                              and not df_input[col].equals(df_output[col])]
    }


@hydra.main(
    config_path=str(PROJECT_ROOT),
    config_name="config",
    version_base=None)
def main(cfg: DictConfig) -> None:
    config_path = PROJECT_ROOT / "config.yaml"

    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"feature_eng_{dt_str}"

    run = None
    try:
        run = wandb.init(
            project=cfg.main.WANDB_PROJECT,
            entity=cfg.main.WANDB_ENTITY,
            job_type="feature_eng",
            name=run_name,
            config=dict(cfg),
            tags=["feature_eng"]
        )
        logger.info("Started WandB run: %s", run_name)

        # Load preprocessed data artifact from W&B
        preprocess_art = run.use_artifact("preprocessed_data:latest")
        with tempfile.TemporaryDirectory() as tmp_dir:
            preprocess_path = preprocess_art.download(root=tmp_dir)
            # Use the same CSV reading parameters as other steps
            data_source_config = OmegaConf.to_container(
                cfg.data_source, resolve=True)
            csv_params = {
                "keep_default_na": data_source_config.get(
                    "keep_default_na", True),
                "na_values": data_source_config.get("na_values", None)
            }
            # Read CSV with config parameters
            df = pd.read_csv(
                os.path.join(preprocess_path, "preprocessed_data.csv"),
                **csv_params
            )

        if df.empty:
            logger.warning("Loaded dataframe is empty.")
            wandb.summary.update({"feature_eng_result": "no_data"})
            return

        logger.info(f"Loaded preprocessed data with shape: {df.shape}")

        # Calculate input data hash for tracking
        input_hash = pd.util.hash_pandas_object(df).sum()
        input_data_hash = f"{input_hash:x}"

        # Log input schema
        input_schema = {col: str(dtype) for col, dtype in df.dtypes.items()}
        with tempfile.NamedTemporaryFile(
                mode='w', suffix='.json', delete=False) as f:
            json.dump(input_schema, f, indent=2)
            input_schema_path = f.name

        # Initialize feature engineer
        engineer = DiabetesFeatureEngineer(str(config_path))
        # Apply feature engineering
        df_processed = engineer.fit_transform(df)
        logger.info(
            f"Feature engineering completed.\
                 Output shape: {df_processed.shape}")

        # Calculate output data hash
        output_hash = pd.util.hash_pandas_object(df_processed).sum()
        output_data_hash = f"{output_hash:x}"

        # Create feature engineering summary
        feature_summary = _create_feature_summary(df, df_processed)
        wandb.summary.update(feature_summary)

        # Log output schema
        output_schema = {col: str(dtype) for col, dtype
                         in df_processed.dtypes.items()}
        with tempfile.NamedTemporaryFile(
                mode='w', suffix='.json', delete=False) as f:
            json.dump(output_schema, f, indent=2)
            output_schema_path = f.name

        # Log output sample artifacts
        if cfg.data_load.get("log_sample_artifacts", True):
            output_sample_tbl = wandb.Table(dataframe=df_processed.head(100))
            wandb.log({"feature_eng_sample_rows": output_sample_tbl})

        if cfg.data_load.get("log_summary_stats", True):
            # Only include numeric columns for summary stats
            numeric_df = df_processed.select_dtypes(include=['number'])
            if not numeric_df.empty:
                output_stats_tbl = wandb.Table(
                    dataframe=numeric_df.describe(
                        include="all").T.reset_index()
                )
                wandb.log({"feature_eng_summary_stats": output_stats_tbl})

        # Save processed data to temporary file and log to W&B
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "engineered_data.csv"
            df_processed.to_csv(tmp_path, index=False)

            if cfg.data_load.get("log_artifacts", True):
                feature_artifact = wandb.Artifact(
                    "engineered_data", type="dataset")
                feature_artifact.add_file(
                    str(tmp_path), name="engineered_data.csv")
                feature_artifact.add_file(
                    input_schema_path, name="input_schema.json")
                feature_artifact.add_file(
                    output_schema_path, name="output_schema.json")
                run.log_artifact(feature_artifact, aliases=["latest"])
                logger.info("Logged engineered data artifact to WandB")

        # Log feature engineering transformation details
        features_config = OmegaConf.to_container(cfg.features, resolve=True)
        transform_details = {
            "feature_engineering_enabled": features_config.get(
                "enabled", True),
            "medication_columns_count": len(features_config.get(
                "medication_columns", [])),
            "icd9_categories_count": len(features_config.get(
                "icd9_categories", {})),
            "value_mappings_count": len(features_config.get(
                "value_mappings", {})),
            "transformers_applied": ["icd9_categorizer", "medication_creator",
                                     "value_mapper", "additional_creator"]
        }

        with tempfile.NamedTemporaryFile(
                mode='w', suffix='.json', delete=False) as f:
            json.dump(transform_details, f, indent=2)
            transform_details_path = f.name

        # Create feature engineering report artifact
        if cfg.data_load.get("log_artifacts", True):
            report_artifact = wandb.Artifact(
                "feature_eng_report", type="report")
            report_artifact.add_file(transform_details_path,
                                     name="feature_eng_config.json")
            run.log_artifact(report_artifact, aliases=["latest"])
            logger.info("Logged feature engineering report to WandB")

        # Log final summary metrics
        wandb.summary.update({
            "feature_eng_result": "success",
            "input_rows": df.shape[0],
            "output_rows": df_processed.shape[0],
            "input_columns": df.shape[1],
            "output_columns": df_processed.shape[1],
            "features_added": df_processed.shape[1] - df.shape[1],
            "input_data_hash": input_data_hash,
            "output_data_hash": output_data_hash
        })

        logger.info("Feature engineering completed successfully")
        logger.info(f"Input shape: {df.shape}")
        logger.info(f"Output shape: {df_processed.shape}")
        logger.info(f"Features added: {df_processed.shape[1] - df.shape[1]}")

    except Exception as e:
        logger.exception("Failed during feature engineering step")
        if run is not None:
            run.alert(title="Feature Engineering Error", text=str(e))
            wandb.summary.update({"feature_eng_result": "failed"})
        raise
    finally:
        if wandb.run is not None:
            wandb.finish()
            logger.info("WandB run finished")


if __name__ == "__main__":
    main()
