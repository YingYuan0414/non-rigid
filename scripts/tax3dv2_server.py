#!/usr/bin/env python
"""
TAX3D server
============

Starts FastAPI on localhost:<port>, keeps the model in VRAM,
waits for JSON messages of the form

    { "fname": "demo_aug_0.npz" }

reads   <work_dir>/data/<fname>
writes  <work_dir>/data/<fname>_pred.npz
writes  <work_dir>/vis/<fname>/...  (if vis enabled)

POST /infer   – returns {"ok": true, "pred_fname": "..._pred.npz"}
POST /shutdown – shuts the process down cleanly
"""
import argparse, json, os, sys, uvicorn, numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException

import hydra
from hydra.core.hydra_config import HydraConfig
import lightning as L
import omegaconf
import torch
import wandb
import os, os.path as osp
import glob

from non_rigid.utils.script_utils import (
    create_model,
    create_datamodule,
    load_checkpoint_config_from_wandb,
)
from non_rigid.metrics.flow_metrics import flow_rmse
from non_rigid.metrics.rigid_metrics import svd_estimation
from non_rigid.utils.pointcloud_utils import expand_pcd, downsample_pcd
from non_rigid.models.gmm_predictor import FrameGMMPredictor
from pytorch3d.transforms import Translate
from tqdm import tqdm
import numpy as np
import open3d as o3d

# -------------------------------------------------------------------------

app = FastAPI()
STATE = {}

# -------------------------------------------------------------------------

def visualize_point_cloud(vis_dict, save_dir, views_to_render=["default", "top", "side", "diagonal"]):
    """Processes and visualizes a single point cloud file."""
    
    # Create the subdirectory based on the input file name
    os.makedirs(save_dir, exist_ok=True)

    anchor_pc = vis_dict["anchor_pc"]
    action_pc = vis_dict["action_pc"]
    initial_action_pc = vis_dict["initial_action_pc"]
    gt_action_pc = vis_dict["gt_action_pc"]

    # Create Open3D point clouds
    action_cloud = o3d.geometry.PointCloud()
    anchor_cloud = o3d.geometry.PointCloud()
    initial_action_cloud = o3d.geometry.PointCloud()
    gt_action_cloud = o3d.geometry.PointCloud()

    action_cloud.points = o3d.utility.Vector3dVector(action_pc)
    anchor_cloud.points = o3d.utility.Vector3dVector(anchor_pc)
    initial_action_cloud.points = o3d.utility.Vector3dVector(initial_action_pc)
    gt_action_cloud.points = o3d.utility.Vector3dVector(gt_action_pc)

    action_cloud.paint_uniform_color([0, 0, 1])  # Blue
    anchor_cloud.paint_uniform_color([1, 0, 0])  # Red
    initial_action_cloud.paint_uniform_color([1, 1, 0])  # Yellow
    gt_action_cloud.paint_uniform_color([0, 1, 0])  # Green

    # Compute bounding box for dynamic camera positioning
    combined_cloud = action_cloud + anchor_cloud + initial_action_cloud + gt_action_cloud
    bounding_box = combined_cloud.get_axis_aligned_bounding_box()
    center = bounding_box.get_center()
    extent = bounding_box.get_extent()

    # Define camera positions for views
    camera_views = {
        "default": center + np.array([0, 0, max(extent) * 2.5]),
        "top": center + np.array([0, max(extent) * 2.5, 0]),
        "side": center + np.array([max(extent) * 2.5, 0, 0]),
        "diagonal": center + np.array([max(extent) * 2.5, max(extent) * 2.5, max(extent) * 2.5]),
    }


    # Render and save each view
    width, height = 1920, 1080  # Resolution
    for view in views_to_render:
        eye = camera_views[view]
        up = [0, 1, 0]

        # Create OffscreenRenderer
        scene = o3d.visualization.rendering.OffscreenRenderer(width, height)
        scene.scene.add_geometry("action", action_cloud, o3d.visualization.rendering.MaterialRecord())
        scene.scene.add_geometry("anchor", anchor_cloud, o3d.visualization.rendering.MaterialRecord())
        scene.scene.add_geometry("initial_action", initial_action_cloud, o3d.visualization.rendering.MaterialRecord())
        scene.scene.add_geometry("gt_action", gt_action_cloud, o3d.visualization.rendering.MaterialRecord())
        scene.scene.camera.look_at(center.tolist(), eye.tolist(), up)

        # Render the image and save it
        image = scene.render_to_image()
        save_path = os.path.join(save_dir, f"{view}_view.png")
        o3d.io.write_image(save_path, image)
        #print(f"Saved {view} view visualization to {save_path}")

        # Explicitly clear the renderer to free up resources
        scene.scene.clear_geometry()
        del scene  # Ensure the renderer is destroyed

def move_to_cpu_nested(d):
    result = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.cpu()
        elif isinstance(v, dict):
            result[k] = move_to_cpu_nested(v)
        else:
            result[k] = v
    return result

def merge_and_move_to_cpu(batch, pred):
    return {
        "batch": move_to_cpu_nested(batch),
        "pred": move_to_cpu_nested(pred)
    }

def log(str):
    out = f"\033[94m[STATUS]  {str}\033[0m"
    print(out)

# -------------------------------------------------------------------------
@torch.no_grad()
@hydra.main(config_path="../configs", config_name="eval_server", version_base="1.3")
def main(cfg):
    #####################################################################
    # Fetch configs
    #####################################################################
    task_overrides = HydraConfig.get().overrides.task
    cfg = load_checkpoint_config_from_wandb(
        cfg, 
        task_overrides, 
        cfg.wandb.entity, 
        cfg.wandb.project, 
        cfg.checkpoint.run_id
    )

    ######################################################################
    # Manually setting eval-specific configs.
    ######################################################################
    cfg.dataset.num_demos = 1
    cfg.inference.batch_size = 1
    cfg.inference.val_batch_size = 1
    cfg.inference.batch_size = 1

    server_cfg = cfg.server
    model_cfg = cfg.model
    dataset_cfg = cfg.dataset
    gmm_cfg = cfg.gmm
    assert server_cfg.work_dir is not None
    print("##################### Diffusion Config #####################")
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
    # Load the GMM frame predictor, if necessary.
    ######################################################################
    if gmm_cfg.gmm is not None:
        # GMM can only be used with noisy goal models.
        if model_cfg.pred_frame != "noisy_goal":
            raise ValueError("GMM can only be used with noisy goal models.")

        gmm_exp_path = os.path.join(
            os.path.expanduser(gmm_cfg.gmm_log_dir),
            f"gmm_{cfg.dataset.name}",
            gmm_cfg.task_name,
            f"epochs={gmm_cfg.gmm}_var={gmm_cfg.var}_unif={gmm_cfg.uniform_loss}_rr={gmm_cfg.regularize_residual}_enc={gmm_cfg.point_encoder}_pn2scale={gmm_cfg.pcd_scale}"
        )

        if os.path.exists(gmm_exp_path):
            print(f"Found GMM experiment directory: {gmm_exp_path}")
        else:
            raise ValueError(f"GMM experiment directory does not exist: {gmm_exp_path}")

        script_dir = os.path.dirname(os.path.abspath(__file__))

        gmm_yaml_path = os.path.join(script_dir, "..", "configs", "model", "df_cross.yaml")
        gmm_model_spec = omegaconf.OmegaConf.load(os.path.abspath(gmm_yaml_path))
        gmm_model_spec.rel_pos = True
        gmm_model_spec.point_encoder = gmm_cfg.point_encoder
        gmm_model_spec.pcd_scale = gmm_cfg.pcd_scale

        print("##################### GMM Config #####################")
        print(
            json.dumps(
                omegaconf.OmegaConf.to_container(gmm_model_spec, resolve=True, throw_on_missing=False),
                sort_keys=True,
                indent=4,
            )
        )
        # Load GMM model config and checkpoint
        #gmm_model_cfg = omegaconf.OmegaConf.load(os.path.join(hydra.utils.get_original_cwd(), "../configs/model/df_cross.yaml"))
        gmm_model = FrameGMMPredictor(gmm_model_spec, device)
        gmm_path = os.path.join(gmm_exp_path, "checkpoints", f"epoch_{gmm_cfg.gmm}.pt")
        gmm_model.load_state_dict(torch.load(gmm_path))
        print(f"Using GMM checkpoint: {gmm_path}")

    else:
        gmm_model = None
        print(f"Using Noisy Oracle")

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
        print(f"Using Wandb checkpoint: {checkpoint_reference}, {model_name}")
    else:
        ckpt_file = checkpoint_reference
        print(f"Using local checkpoint: {ckpt_file}")
    # Load the network weights.
    ckpt = torch.load(ckpt_file, map_location=device)
    network.load_state_dict(
        {k.partition(".")[2]: v for k, v, in ckpt["state_dict"].items() if k.startswith("network.")}
    )
    # set model to eval mode
    network.eval()
    model.eval()
    model.to(device)

    val_dataloader, test_dataloader = datamodule.val_dataloader()

    test_dataloader.dataset.set_eval_mode(True)
    test_dataloader.dataset.demo_files = [None]
    
    #################################################################################################################################
    STATE.update(dict(cfg=cfg, dataloader=test_dataloader,
                      network=network, model=model, gmm_model=gmm_model,
                      device=device, work_dir=Path(server_cfg.work_dir)))


    log_str = f"Model initialization succeeded. Working on root directory: {server_cfg.work_dir}"
    log(log_str)

    uvicorn.run(app, host="127.0.0.1", port=server_cfg.port, log_level="info")


# -------------------------------------------------------------------------
"""
Endpoint: /infer (POST)

Description:
    Perform inference on a given input file, save the predicted output, and optionally visualize the results.
    This endpoint expects a JSON payload with the necessary information to locate the input file,
    perform model inference, and handle outputs.

Expected Payload (JSON):
    {
        "fname": (str) Filename of the input .npz file, relative to `STATE["work_dir"]/data/`.
        "save_pred": (bool, optional) Whether to save prediction results. 
        "save_vis": (bool, optional) Whether to save visualization images.
    }

Returns:
    {
        "ok": True,
        "pred_fname": (str) Name of the saved prediction .npz file
    }

Notes:
    - The input file must already exist under `STATE["work_dir"]/data/`.
    - Visualization images are saved under `STATE["work_dir"]/vis/{input_filename_stem}/`.
    - `STATE` must contain initialized `work_dir`, `dataloader`, `model`, `gmm_model` (optional), and `cfg`.
"""
@app.post("/infer")
def infer(payload: dict):
    fname = payload.get("fname")
    save_pred = payload.get("save_pred")
    save_vis = payload.get("save_vis")
    infer_name = payload.get("infer_name")

    if fname is None:
        raise HTTPException(400, "missing fname")
    work_dir = STATE["work_dir"]
    in_f  = work_dir / "data" / fname
    out_f = in_f.with_stem(in_f.stem + "_pred")
    if not in_f.is_file():
        raise HTTPException(404, f"{fname} not found")

    log_str = f"Received inference request for: {fname}"
    log(log_str)

    dataloader  = STATE["dataloader"]
    model  = STATE["model"]
    gmm_model  = STATE["gmm_model"]
    cfg = STATE["cfg"]

    if infer_name == "datamodule":
        log_str = f"Inference begin : infer method using datamodule"
        log(log_str)

        dataloader.dataset.demo_files = [str(in_f)]
        dataloader.dataset.set_eval_mode(True)
        assert len(dataloader.dataset) == 1

        with torch.no_grad():
            for batch in tqdm(dataloader):
                # Generate predictions.
                if gmm_model is not None:
                    # Yucky hack for GMM; need to expand point clouds first for WTA GMM samples.
                    pred_batch = {key: expand_pcd(value, cfg.inference.num_gmm_trials) for key, value in batch.items()}
                    pred_batch = model.update_batch_frames(pred_batch, update_labels=True, gmm_model=gmm_model)
                    batch = model.update_batch_frames(batch, update_labels=True)
                    pred_dict = model.predict(pred_batch, cfg.inference.num_trials, progress=True, full_prediction=True)
                else:
                    pred_batch = model.update_batch_frames(batch, update_labels=True)
                    pred_dict = model.predict(pred_batch, cfg.inference.num_trials, progress=True, full_prediction=True)
                break   # there should only be one file

    elif infer_name == "dataset":
        log_str = f"Inference begin : infer method using dataset"
        log(log_str)

        dataset = dataloader.dataset
        dataset.demo_files = [str(in_f)]
        batch = dataset.__getitem__(0)
        batch = {k: v.unsqueeze(0) for k, v in batch.items()}

        with torch.no_grad():
            if gmm_model is not None:
                # Yucky hack for GMM; need to expand point clouds first for WTA GMM samples.
                pred_batch = {key: expand_pcd(value, cfg.inference.num_gmm_trials) for key, value in batch.items()}
                pred_batch = model.update_batch_frames(pred_batch, update_labels=True, gmm_model=gmm_model)
                batch = model.update_batch_frames(batch, update_labels=True)
                pred_dict = model.predict(pred_batch, cfg.inference.num_trials, progress=True, full_prediction=True)
            else:
                pred_batch = model.update_batch_frames(batch, update_labels=True)
                pred_dict = model.predict(pred_batch, cfg.inference.num_trials, progress=True, full_prediction=True)
    
    elif infer_name == "direct":
        demo = np.load(in_f, allow_pickle=True)

        parent_start_pcd = demo['multi_obj_start_pcd'].item()['parent']
        child_start_pcd = demo['multi_obj_start_pcd'].item()['child']

        anchor_pc = torch.from_numpy(parent_start_pcd * cfg.dataset.pcd_scale_factor)
        action_pc = torch.from_numpy(child_start_pcd * cfg.dataset.pcd_scale_factor)

        # prob want to convert anchor/action_pc to tensor?
        sample_size_action = cfg.dataset.sample_size_action
        sample_size_anchor = cfg.dataset.sample_size_anchor

        action_seg = torch.zeros_like(action_pc[:, 0]).int()
        anchor_seg = torch.ones_like(anchor_pc[:, 0]).int()

        # downsample action
        
        if action_pc.shape[0] > sample_size_action:
            action_pc, action_pc_indices = downsample_pcd(action_pc.unsqueeze(0), sample_size_action, type='fps')
            action_pc = action_pc.squeeze(0)
            action_seg = action_seg[action_pc_indices.squeeze(0)]

        # downsample anchor
        if anchor_pc.shape[0] > sample_size_anchor:
            anchor_pc, anchor_pc_indices = downsample_pcd(anchor_pc.unsqueeze(0), sample_size_anchor, type='fps')
            anchor_pc = anchor_pc.squeeze(0)
            anchor_seg = anchor_seg[anchor_pc_indices.squeeze(0)]
        
        action_pc = action_pc.to(torch.float32)
        anchor_pc = anchor_pc.to(torch.float32)

        scene_center = torch.cat([action_pc, anchor_pc], dim=0).mean(axis=0)
        anchor_pc = anchor_pc - scene_center
        action_pc = action_pc - scene_center

        T_goal2world = Translate(scene_center.unsqueeze(0))
        T_action2world = Translate(scene_center.unsqueeze(0))

        batch = {}

        batch["pc_action"] = action_pc.unsqueeze(0) # Action points in the action frame
        batch["pc_anchor"] = anchor_pc.unsqueeze(0) # Anchor points in the scene frame
        batch["seg"] = action_seg.unsqueeze(0)
        batch["seg_anchor"] = anchor_seg.unsqueeze(0)
        batch["T_goal2world"] = T_goal2world.get_matrix() # Transform from goal action frame to world frame
        batch["T_action2world"] = T_action2world.get_matrix() # Transform from action frame to world frame

        with torch.no_grad():
            if gmm_model is not None:
                # Yucky hack for GMM; need to expand point clouds first for WTA GMM samples.
                pred_batch = {key: expand_pcd(value, cfg.inference.num_gmm_trials) for key, value in batch.items()}
                pred_batch = model.update_batch_frames(pred_batch, update_labels=False, gmm_model=gmm_model)
                #batch = model.update_batch_frames(batch, update_labels=False)
                pred_dict = model.predict(pred_batch, cfg.inference.num_trials, progress=True, full_prediction=True)
            else:
                pred_batch = model.update_batch_frames(batch, update_labels=False)
                pred_dict = model.predict(pred_batch, cfg.inference.num_trials, progress=True, full_prediction=True)

    else:
        raise HTTPException(404, f"{infer_name} is not supported")


    pred_T = pred_dict["pred_T"].cpu().numpy()
    pred_frame_world = pred_dict["pred_frame_world"].cpu().numpy() / cfg.dataset.pcd_scale_factor
    pred_point_world = pred_dict["point"]["pred_world"].cpu().numpy() / cfg.dataset.pcd_scale_factor
    init_action_world = pred_dict["init_action_world"][0].cpu().numpy() / cfg.dataset.pcd_scale_factor

    log_str = f"Inference done: Pred_T shape {pred_T.shape}"
    log(log_str)

    # save pred
    
    if save_pred:
        merged_dict = merge_and_move_to_cpu(pred_batch, pred_dict)
        merged_dict['cfg'] = {
            'pcd_scale_factor': cfg.dataset.pcd_scale_factor
        }
        np.savez(out_f, **merged_dict)

        log_str = f"Inference saved: {out_f}"
        log(log_str)

    # save visualisation
    if save_vis:
        vis_dir = work_dir / "vis" / in_f.stem
        vis_dir.mkdir(parents=True, exist_ok=True)
        visualize_point_cloud({
            "anchor_pc":          pred_batch["pc_anchor"][0].cpu(),
            "action_pc":          pred_dict["point"]["pred"][0].cpu(),
            "initial_action_pc":  pred_batch["pc_action"][0].cpu(),
            "gt_action_pc":       pred_batch["pc"][0].cpu(),
        }, vis_dir)

        log_str = f"Visualization saved: {vis_dir}"
        log(log_str)

    return {"ok": True, 
            "pred_T": pred_T.tolist(), 
            "pred_p": pred_point_world.tolist(), 
            "pred_f": pred_frame_world.tolist(),
            "init_action":init_action_world.tolist()}

# -------------------------------------------------------------------------
@app.post("/shutdown")
def shutdown():
    os._exit(0)

if __name__ == "__main__":
    main()
