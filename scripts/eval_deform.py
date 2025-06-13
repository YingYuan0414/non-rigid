import hydra
from hydra.core.hydra_config import HydraConfig
import lightning as L
import json
import omegaconf
import torch
import wandb
import os

from non_rigid.utils.script_utils import (
    create_model,
    create_datamodule,
    load_checkpoint_config_from_wandb,
)
from non_rigid.metrics.flow_metrics import flow_rmse
from non_rigid.utils.pointcloud_utils import expand_pcd
from non_rigid.models.gmm_predictor import FrameGMMPredictor
from tqdm import tqdm
import numpy as np

import rpad.visualize_3d.plots as vpl


@torch.no_grad()
@hydra.main(config_path="../configs", config_name="eval_deform", version_base="1.3")
def main(cfg):
    task_overrides = HydraConfig.get().overrides.task
    cfg = load_checkpoint_config_from_wandb(
        cfg, 
        task_overrides, 
        cfg.wandb.entity, 
        cfg.wandb.project, 
        cfg.checkpoint.run_id
    )
    print(
        json.dumps(
            omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False),
            sort_keys=True,
            indent=4,
        )
    )
    ######################################################################
    # Torch settings.
    ######################################################################

    # Make deterministic + reproducible.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Since most of us are training on 3090s+, we can use mixed precision.
    torch.set_float32_matmul_precision("medium")

    # Global seed for reproducibility.
    L.seed_everything(42)

    device = f"cuda:{cfg.resources.gpus[0]}"

    ######################################################################
    # Manually setting eval-specific configs.
    ######################################################################
    # Using a custom cloth-specific batch size, to allow for simultaneous evaluation 
    # of RMSE, coverage, and precision.
    if cfg.dataset.hole == "single":
        bs = 1
    elif cfg.dataset.hole == "double":
        bs = 2
    else:
        raise ValueError(f"Unknown hole type: {cfg.dataset.hole}.")
    bs *= cfg.dataset.num_anchors

    cfg.inference.batch_size = bs
    cfg.inference.val_batch_size = bs

    ######################################################################
    # Load the GMM frame predictor, if necessary.
    ######################################################################
    if cfg.gmm is not None:
        # GMM can only be used with noisy goal models.
        if cfg.model.pred_frame != "noisy_goal":
            raise ValueError("GMM can only be used with noisy goal models.")
        
        # Grab config file from saved run.
        exp_name = os.path.join(
            os.path.expanduser(cfg.gmm_log_dir),
            cfg.gmm,
        )
        if not os.path.exists(exp_name):
            raise ValueError(f"Experiment directory {exp_name} does not exist - train this model first.")
        gmm_cfg = omegaconf.OmegaConf.load(os.path.join(exp_name, "config.yaml"))
        
        # Creating GMM network.
        network, _ = create_model(gmm_cfg)
        gmm_model = FrameGMMPredictor(network, gmm_cfg.model, device)
        gmm_model.load_state_dict(
            torch.load(
                os.path.join(exp_name, "checkpoints", f"epoch_{gmm_cfg.epochs}.pt")
            )
        )
        gmm_model.eval()
    else:
        gmm_model = None

    ######################################################################
    # Create the datamodule. This is just to initialize the datasets - we are
    # not going to use the dataloaders, because we need to manually downsample 
    # and batch.
    ######################################################################
    cfg, datamodule = create_datamodule(cfg)

    ######################################################################
    # Create the network(s) which will be evaluated (same as training).
    # You might want to put this into a "create_network" function
    # somewhere so train and eval can be the same.
    #
    # We'll also load the weights.
    ######################################################################

    # Model architecture is dataset-dependent, so we have a helper
    # function to create the model (while separating out relevant vals).
    network, model = create_model(cfg)    

    # get checkpoint file (for now, this does not log a run)
    checkpoint_reference = cfg.checkpoint.reference
    if checkpoint_reference.startswith(cfg.wandb.entity):
        api = wandb.Api()
        artifact_dir = cfg.wandb.artifact_dir
        artifact = api.artifact(checkpoint_reference, type="model")

        if cfg.checkpoint.alias == "v0":
            model_name = "model.ckpt"
        elif cfg.checkpoint.alias == "monitor":
            # getting artifact names, and sanity checking monitor name
            artifact_file_names = [f.name for f in artifact.files()]
            monitor_name = cfg.checkpoint.monitor_name
            if not isinstance(monitor_name, str):
                raise ValueError(f"Invalid monitor name: {monitor_name}. Must be a string.")
            
            # searching for checkpoints with exact monitor name - should only be one for now.
            valid_artifact_file_names = [f for f in artifact_file_names if 
                                         f.split("-")[2].split("=")[0] == monitor_name]
            if len(valid_artifact_file_names) == 0:
                raise ValueError(f"Could not find any files with monitor name: {monitor_name}.")
            elif len(valid_artifact_file_names) > 1:
                raise ValueError(f"Found multiple files with monitor name: {monitor_name}.")
            else:
                model_name = valid_artifact_file_names[0]
        else:
            raise ValueError(f"Invalid checkpoint alias: {cfg.checkpoint.alias}.")
        ckpt_file = artifact.get_path(model_name).download(root=artifact_dir)
    else:
        ckpt_file = checkpoint_reference
    # Load the network weights.
    ckpt = torch.load(ckpt_file, map_location=device)
    network.load_state_dict(
        {k.partition(".")[2]: v for k, v, in ckpt["state_dict"].items() if k.startswith("network.")}
    )
    # set model to eval mode
    network.eval()
    model.eval()

    ######################################################################
    # Helper function to run evals for a given dataset.
    ######################################################################
    def run_eval(dataset, model):
        num_samples = cfg.inference.num_wta_trials # // bs
        num_batches = len(dataset) // bs
        eval_keys = ["pc_action", "pc_anchor", "pc", "flow", "seg", "seg_anchor", "T_action2world", "T_goal2world"]
        if cfg.model.pred_frame == "noisy_goal":
            eval_keys.append("noisy_goal")
            
        rmse = []
        coverage = []
        precision = []

        for i in tqdm(range(num_batches)):
            batch_list = []

            # get first item in batch, and keep downsampling indices
            item = dataset.__getitem__(i * bs, return_indices=True)
            downsample_indices = {
                "action_pc_indices": item["action_pc_indices"],
                "anchor_pc_indices": item["anchor_pc_indices"],
            }
            batch_list.append({key: item[key] for key in eval_keys})

            # get the rest of the batch
            for j in range(1, bs):
                item = dataset.__getitem__(i * bs + j, use_indices=downsample_indices)
                batch_list.append({key: item[key] for key in eval_keys})

            # convert to batch
            batch = {key: torch.stack([item[key] for item in batch_list]) for key in eval_keys}

            # Generate predictions.
            if gmm_model is not None:
                # Yucky hack for GMM; need to expand point clouds first for WTA GMM samples.
                gmm_batch = {key: expand_pcd(value, num_samples) for key, value in batch.items()}
                gmm_batch = model.update_batch_frames(gmm_batch, update_labels=True, gmm_model=gmm_model)
                batch = model.update_batch_frames(batch, update_labels=True)
                pred_dict = model.predict(gmm_batch, 1, progress=False, full_prediction=True)
            else:
                batch = model.update_batch_frames(batch, update_labels=True)
                pred_dict = model.predict(batch, num_samples, progress=False, full_prediction=True)
            pred_point_world = pred_dict["point"]["pred_world"]


            batch_rmse = torch.zeros(bs, cfg.inference.num_wta_trials * bs)

            for j in range(bs):
                # expand ground truth pc to compute RMSE for cloth-specific sample
                gt_pc_world = batch["pc_world"][j].unsqueeze(0).to(device)
                seg = batch["seg"][j].unsqueeze(0).to(device)
                gt_pc_world = expand_pcd(gt_pc_world, num_samples * bs)
                seg = expand_pcd(seg, num_samples * bs)

                seg = seg == 0
                batch_rmse[j] = flow_rmse(pred_point_world, gt_pc_world, mask=True, seg=seg)

            # computing precision and coverage
            batch_precision = torch.min(batch_rmse, dim=0).values
            batch_coverage = torch.min(batch_rmse, dim=1).values

            # update dataset-wide metrics
            rmse.append(batch_rmse.mean().item())
            coverage.append(batch_coverage.mean().item())
            precision.append(batch_precision.mean().item())
            
        rmse = np.mean(rmse)
        coverage = np.mean(coverage)
        precision = np.mean(precision)
        return rmse, coverage, precision
    

    ######################################################################
    # Run the model on the train/val/test sets.
    ######################################################################
    model.to(device)

    train_rmse, train_coverage, train_precision = run_eval(datamodule.train_dataset, model)
    val_rmse, val_coverage, val_precision = run_eval(datamodule.val_dataset, model)
    # val_ood_rmse, val_ood_coverage, val_ood_precision = run_eval(datamodule.val_ood_dataset, model)

    print(f"Train RMSE: {train_rmse}, Coverage: {train_coverage}, Precision: {train_precision}")
    print(f"Val RMSE: {val_rmse}, Coverage: {val_coverage}, Precision: {val_precision}")
    # print(f"Val OOD RMSE: {val_ood_rmse}, Coverage: {val_ood_coverage}, Precision: {val_ood_precision}")

if __name__ == "__main__":
    main()