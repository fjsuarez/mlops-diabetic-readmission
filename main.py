import mlflow
import tempfile
import os
import hydra
from omegaconf import DictConfig
from dotenv import load_dotenv
from datetime import datetime
import wandb

load_dotenv()

PIPELINE_STEPS = [
    "data_load",
    "data_validation",
    "preprocess",
    "features"
]

# Only these steps accept Hydra overrides via MLflow parameters
STEPS_WITH_OVERRIDES = {"model"}


@hydra.main(config_name="config", config_path=".", version_base=None)
def main(cfg: DictConfig):
    os.environ["WANDB_PROJECT"] = cfg.main.WANDB_PROJECT
    os.environ["WANDB_ENTITY"] = cfg.main.WANDB_ENTITY

    run_name = f"orchestrator_{datetime.now():%Y%m%d_%H%M%S}"
    run = wandb.init(
        project=cfg.main.WANDB_PROJECT,
        entity=cfg.main.WANDB_ENTITY,
        job_type="orchestrator",
        name=run_name,
    )
    print(f"Started WandB run: {run.name}")

    steps_raw = cfg.main.steps
    active_steps = [s.strip() for s in steps_raw.split(",") if s.strip()] \
        if steps_raw != "all" else PIPELINE_STEPS

    hydra_override = cfg.main.hydra_options if hasattr(
        cfg.main, "hydra_options") else ""

    with tempfile.TemporaryDirectory():
        for step in active_steps:
            step_dir = os.path.join(
                hydra.utils.get_original_cwd(), "src", step)

            params = {}
            if hydra_override and step in STEPS_WITH_OVERRIDES:
                params["hydra_options"] = hydra_override

            print(f"Running step: {step} (W&B run: {run.name})")
            mlflow.run(step_dir, "main", parameters=params)

    wandb.finish()


if __name__ == "__main__":
    main()
