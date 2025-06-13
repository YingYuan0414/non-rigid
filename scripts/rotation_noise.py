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
from non_rigid.metrics.rigid_metrics import svd_estimation
from non_rigid.utils.pointcloud_utils import expand_pcd
from tqdm import tqdm
import numpy as np

from plotly import graph_objects as go

@torch.no_grad()
@hydra.main(config_path="../configs", config_name="eval", version_base="1.3")
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
    model.to(device)

    ######################################################################
    # Helper function to sample rotation errors for forward process.
    ######################################################################
    def forward_rotation_noise(dataset, model):
        num_samples_per_timestep = 20
        diffusion = model.diffusion
        eval_keys = ["pc_action", "pc_anchor", "pc", "flow", "seg", "seg_anchor", "T_action2world", "T_goal2world"]
        if cfg.model.pred_frame == "noisy_goal":
            eval_keys.append("noisy_goal")
        
        rotation_errs_forward = torch.zeros((diffusion.num_timesteps, len(dataset), num_samples_per_timestep)).to(device)

        for i in tqdm(range(len(dataset))):
            # prepare data batch
            item = dataset[i]
            batch = [{key: item[key].to(device) for key in eval_keys}]
            batch = {key: torch.stack([item[key] for item in batch]) for key in eval_keys}
            batch = {key: expand_pcd(value, num_samples_per_timestep) for key, value in batch.items()}
            batch = model.update_batch_frames(batch, update_labels=True)

            #######################################################
            # Forward process.
            ########################################################
            ground_truth = batch[model.label_key].permute(0, 2, 1)
            xr_start = ground_truth.mean(dim=-1, keepdim=True)
            xs_start = ground_truth - xr_start

            for j in tqdm(range(diffusion.num_timesteps)):
                # Sampling noise.
                # noise_r = torch.randn_like(xr_start)
                noise_s = torch.randn_like(xs_start)

                if diffusion.rotation_noise_scale:
                    N, C, P = xs_start.shape  # e.g. [batch_size, 3, num_points]

                    # 1. Sample random rotation axes and normalize them.
                    random_axis = torch.randn(N, 3, device=xs_start.device)
                    random_axis = random_axis / random_axis.norm(dim=1, keepdim=True)

                    # 2. Sample random rotation angles (in degrees) and convert to radians.
                    random_angle = torch.randn(N, device=xs_start.device) * diffusion.rotation_noise_scale
                    random_angle_rad = random_angle * (np.pi / 180)

                    # 3. Compute sin and cos for each angle.
                    sin_theta = torch.sin(random_angle_rad)
                    cos_theta = torch.cos(random_angle_rad)
                    one_minus_cos = 1 - cos_theta

                    # 4. Extract axis components.
                    x = random_axis[:, 0]
                    y = random_axis[:, 1]
                    z = random_axis[:, 2]
                    zeros = torch.zeros_like(x)

                    # 5. Construct the skew-symmetric cross-product matrices for each axis.
                    #    Each K is of shape (3,3), and K will have shape (N, 3, 3)
                    K = torch.stack([torch.stack([zeros, -z, y], dim=1),
                                torch.stack([z, zeros, -x], dim=1),
                                torch.stack([-y, x, zeros], dim=1)], dim=1)

                    # 6. Compute K squared (batched matrix multiplication)
                    K2 = torch.bmm(K, K)

                    # 7. Create the identity matrix for each batch element.
                    I = torch.eye(3, device=xs_start.device).unsqueeze(0).repeat(N, 1, 1)

                    # 8. Compute the rotation matrices using the Rodrigues formula:
                    #    R = I + sin(theta)*K + (1-cos(theta))*(K^2)
                    R = I + sin_theta.view(N, 1, 1) * K + one_minus_cos.view(N, 1, 1) * K2

                    # 9. Apply the rotation matrices to the point clouds.
                    #    x_start: [N, 3, P] -> rotated_pc: [N, 3, P]
                    rotated_pc = torch.bmm(R, xs_start)

                    # 10. The rotation noise is the difference between the rotated and original points.
                    rotation_noise = rotated_pc - xs_start
                    noise_s = noise_s + rotation_noise

                # Forward process.
                # xr_t = diffusion.q_sample(xr_start, t=torch.tensor([j]*num_samples_per_timestep).to(device), noise=noise_r)
                xs_t = diffusion.q_sample(xs_start, t=torch.tensor([j]*num_samples_per_timestep).to(device), noise=noise_s)
                translation_errs, rotation_errs = svd_estimation(
                    xs_start.permute(0, 2, 1), 
                    xs_t.permute(0, 2, 1), 
                    return_magnitude=True
                )
                rotation_errs_forward[j, i] = rotation_errs

        np.save(f"/home/jacinto/data/rotation_noise/{cfg.checkpoint.run_id}_forward.npy", rotation_errs_forward.cpu().numpy())

    ######################################################################
    # Helper function to compute rotation errors for reverse process.
    ######################################################################
    def reverse_rotation_noise(dataset, model):
        num_samples = 20
        eval_keys = ["pc_action", "pc_anchor", "pc", "flow", "seg", "seg_anchor", "T_action2world", "T_goal2world"]
        if cfg.model.pred_frame == "noisy_goal":
            eval_keys.append("noisy_goal")
        
        rotation_errs_reverse = torch.zeros((model.diffusion.num_timesteps + 1, len(dataset), num_samples)).to(device)
        
        for i in tqdm(range(len(dataset))):
            # prepare data batch
            item = dataset[i]
            batch = [{key: item[key].to(device) for key in eval_keys}]
            batch = {key: torch.stack([item[key] for item in batch]) for key in eval_keys}
            batch = {key: expand_pcd(value, num_samples) for key, value in batch.items()}
            batch = model.update_batch_frames(batch, update_labels=True)

            #######################################################
            # Reverse process.
            ########################################################
            ground_truth = batch[model.label_key]
            pred_frame = batch["pred_frame"].clone()
            pred_frame = expand_pcd(pred_frame, 1)
            bs, sample_size = batch["pc_action"].shape[:2]
            model_kwargs = model.get_model_kwargs(batch, 1)
            z_s = torch.randn(bs, 3, sample_size).to(device)
            z_r = torch.randn(bs, 3, 1).to(device)

            final_dict, results = model.diffusion.p_sample_loop(
                model.network,
                z_r.shape,
                z_s.shape,
                z_r,
                z_s,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=True,
                device=device,
            )
            results_s = [res["sample_s"].permute(0, 2, 1) for res in results]
            rotation_errs = [svd_estimation(ground_truth, res, return_magnitude=True)[1] for res in results_s]
            rotation_errs = torch.stack(rotation_errs, dim=0)
            rotation_errs_reverse[:, i] = rotation_errs
        
        np.save(f"/home/jacinto/data/rotation_noise/{cfg.checkpoint.run_id}_reverse.npy", rotation_errs_reverse.cpu().numpy())

    ######################################################################
    # Helper function to visualize rotation errors.
    ######################################################################
    def visualize_rotation_errors(rotation_errs, title):
        forward_rots = np.load(f"/home/jacinto/data/rotation_noise/{cfg.checkpoint.run_id}_forward.npy")
        reverse_rots = np.load(f"/home/jacinto/data/rotation_noise/{cfg.checkpoint.run_id}_reverse.npy")

        forward_rots = forward_rots.reshape(forward_rots.shape[0], -1)
        reverse_rots = reverse_rots.reshape(reverse_rots.shape[0], -1)[:100, :]

        forward_rots_mean = np.mean(forward_rots, axis=1)
        reverse_rots_mean = np.flip(np.mean(reverse_rots, axis=1))
        forward_rots_std = np.std(forward_rots, axis=1)
        reverse_rots_std = np.flip(np.std(reverse_rots, axis=1))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=np.arange(len(forward_rots_mean)),
            y=forward_rots_mean,
            mode='lines+markers',
            name='Forward Process Mean',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=np.arange(len(forward_rots_std)),
            y=forward_rots_std,
            mode='lines+markers',
            name='Forward Process Std',
            line=dict(color='blue', dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=np.arange(len(reverse_rots_mean)),
            y=reverse_rots_mean,
            mode='lines+markers',
            name='Reverse Process Mean',
            line=dict(color='red')
        ))
        fig.add_trace(go.Scatter(
            x=np.arange(len(reverse_rots_std)),
            y=reverse_rots_std,
            mode='lines+markers',
            name='Reverse Process Std',
            line=dict(color='red', dash='dash')
        ))
        fig.update_layout(
            title=f"Rotation Errors for {title}",
            xaxis_title="Time Step",
            yaxis_title="Rotation Error (degrees)",
            legend_title="Legend",
            template="plotly_white"
        )
        fig.show()
        breakpoint()

    # forward_rotation_noise(datamodule.val_dataset, model)
    # reverse_rotation_noise(datamodule.val_dataset, model)
    visualize_rotation_errors(datamodule.val_dataset, cfg.checkpoint.run_id)

if __name__ == "__main__":
    main()