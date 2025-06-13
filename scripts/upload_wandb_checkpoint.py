import wandb
import os
import yaml
import subprocess
import argparse


# Argument parser for command line arguments.
parser = argparse.ArgumentParser(description="Sync and upload WandB checkpoints.")
parser.add_argument(
    "--run_dir",
    type=str,
    help="Directory containing the WandB run and checkpoints.",
)


# run_dir = "/home/eycai/tax3d_checkpoints/21-28-02"
run_dir = parser.parse_args().run_dir
# Ensure the run directory exists.
if not os.path.exists(run_dir):
    raise ValueError(f"Run directory {run_dir} does not exist.")
wandb_dir = os.path.join(run_dir, "wandb", "latest-run")

#########################################################################
# SYNCING CRASHED WANDB RUN.
#########################################################################
subprocess.run(["wandb", "sync", wandb_dir], check=True)


#########################################################################
# RESUMING WANDB RUN TO UPLOAD CHECKPOINTS.
#########################################################################
# Extracting wandb run information.
run_cfg = yaml.safe_load(open(os.path.join(wandb_dir, "files", "config.yaml"), "r"))
entity = run_cfg["wandb"]["value"]["entity"]
project = run_cfg["wandb"]["value"]["project"]
group = run_cfg["wandb"]["value"]["group"]

# Extracting wandb run ID.
wandb_dir_files = [f for f in os.listdir(wandb_dir) if f.startswith("run-")]
if not wandb_dir_files:
    raise ValueError("No wandb run directory found in the specified path.")
elif len(wandb_dir_files) > 1:
    raise ValueError("Multiple wandb run directories found. Please specify the correct one.")
run_id = wandb_dir_files[0].split("-")[-1].split(".")[0]

print("Uploading checkpoints for run ID:", run_id)
run = wandb.init(
    id=run_id,
    entity=entity,
    project=project,
    group=group,
    resume="allow",
)

#########################################################################
# UPLOADING MODEL CHECKPOINTS.
#########################################################################
checkpoint_dir = os.path.join(run_dir, "checkpoints")
api = wandb.Api()
model_name = f"model-{run_id}"

# Log final checkpoint.
try:
    artifact = api.artifact(f"{entity}/{project}/{model_name}:v0")
    print("Last checkpoint already exists in wandb, skipping upload.")
except wandb.errors.CommError:
    last_artifact = wandb.Artifact(f"model-{run_id}", type="model")
    last_artifact.add_file(os.path.join(checkpoint_dir, "last.ckpt"), name="model.ckpt")
    run.log_artifact(last_artifact, aliases=["best", "v0"])

# Log monitor-based checkpoints.
try:
    artifact = api.artifact(f"{entity}/{project}/{model_name}:monitor")
    print("Monitor checkpoints already exist in wandb, skipping upload.")
except wandb.errors.CommError:
    monitors = ["val_rmse_wta_0", "val_rmse_0"]
    monitor_artifact = wandb.Artifact(f"model-{run_id}", type="model")
    for file in os.listdir(checkpoint_dir):
        if file.endswith(".ckpt"):
            # check if metric name is in monitors
            metric_name = file.split("-")[-1].split("=")[0]
            if metric_name in monitors:
                monitor_artifact.add_file(os.path.join(checkpoint_dir, file))
    run.log_artifact(monitor_artifact, aliases=["latest", "monitor", "v1"])

run.finish()