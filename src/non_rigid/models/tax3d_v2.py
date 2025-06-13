from typing import Any, Dict

import lightning as L
import numpy as np
import torch
import wandb
from diffusers import get_cosine_schedule_with_warmup
from pytorch3d.transforms import Transform3d
from torch import nn, optim

from non_rigid.metrics.error_metrics import get_pred_pcd_rigid_errors
from non_rigid.metrics.flow_metrics import flow_cos_sim, flow_rmse, pc_nn
from non_rigid.metrics.rigid_metrics import svd_estimation, translation_err, rotation_err
from non_rigid.models.dit.diffusion import create_diffusion_mu, create_diffusion_ddrd_separate
from non_rigid.models.dit.models import (
    TAX3Dv2_MuFrame_DiT,
    TAX3Dv2_FixedFrame_Token_DiT,
)
from non_rigid.utils.logging_utils import viz_predicted_vs_gt
from non_rigid.utils.pointcloud_utils import expand_pcd

import rpad.visualize_3d.plots as vpl


def TAX3Dv2_MuFrame_DiT_xS(**kwargs):
    return TAX3Dv2_MuFrame_DiT(depth=5, hidden_size=128, num_heads=4, **kwargs)

def TAX3Dv2_FixedFrame_DiT_xS(**kwargs):
    return TAX3Dv2_FixedFrame_Token_DiT(depth=5, hidden_size=128, num_heads=4, **kwargs)

DiT_models = {
    "TAX3Dv2_MuFrame_DiT_xS": TAX3Dv2_MuFrame_DiT_xS,
    "TAX3Dv2_FixedFrame_DiT_xS": TAX3Dv2_FixedFrame_DiT_xS,
}


def get_model(model_cfg):
    # TODO: move fixed/mu frame to model config as a flag
    if model_cfg.frame_type == "fixed":
        model_name = "TAX3Dv2_FixedFrame_DiT_xS"
    elif model_cfg.frame_type == "mu":
        model_name = "TAX3Dv2_MuFrame_DiT_xS"
    else:
        raise ValueError("Choose model_take from [\"tax3dv2_muframe\", \"tax3dv2_fixedframe\"]")
    return DiT_models[model_name]


class TAX3Dv2Network(nn.Module):
    """
    Network containing the specified Diffusion Transformer architecture.
    """
    def __init__(self, model_cfg=None):
        super().__init__()
        self.dit = get_model(model_cfg)(
            in_channels=model_cfg.in_channels,
            learn_sigma=model_cfg.learn_sigma,
            model_cfg=model_cfg,
        )
    
    def forward(self, xr_t, xs_t, t, **kwargs):
        return self.dit(xr_t, xs_t, t, **kwargs)
    


class TAX3Dv2BaseModule(L.LightningModule):
    """
    Generalized Dense Displacement Diffusion (DDD) module that handles model training, inference, 
    evaluation, and visualization. This module is inherited and overriden by scene-level and 
    object-centric modules.
    """
    def __init__(self, network, cfg) -> None:
        super().__init__()
        self.network = network
        self.mode = cfg.mode # train or eval
        self.model_cfg = cfg.model
        self.dataset_cfg = cfg.dataset
        self.wandb_cfg = cfg.wandb
        self.prediction_type = self.model_cfg.type # flow or point
        self.pred_frame = self.model_cfg.pred_frame
        self.noisy_goal_scale = self.model_cfg.noisy_goal_scale
        # self.model_name = self.model_cfg.name

        # prediction type-specific processing
        # TODO: eventually, this should be removed by updating dataset to use "point" instead of "pc"
        if self.prediction_type == "flow":
            self.label_key = "flow"
        elif self.prediction_type == "point":
            self.label_key = "pc"
        else:
            raise ValueError(f"Invalid prediction type: {self.prediction_type}")
        
        if self.pred_frame == "noisy_goal":
            assert self.model_cfg.noisy_goal_scale > 0.0
        elif self.pred_frame == "anchor_center":
            assert self.model_cfg.noisy_goal_scale == 0.0
        else:
            raise ValueError(f"Invalid prediction type: {self.pred_frame}")

        # For now, model cannot use relative position embedding and scene-as-anchor.
        if self.model_cfg.rel_pos and self.model_cfg.scene_anchor:
            raise ValueError("Relative position embedding and scene-as-anchor are not compatible.")

        # Model cannot have both object and scene scale
        if self.model_cfg.object_scale is not None and self.model_cfg.scene_scale is not None:
            raise ValueError("Model cannot have both object and scene scale.")
        self.object_scale = self.model_cfg.object_scale
        self.scene_scale = self.model_cfg.scene_scale

        # mode-specific processing
        if self.mode == "train":
            self.run_cfg = cfg.training
            # training-specific params
            self.lr = self.run_cfg.lr
            self.weight_decay = self.run_cfg.weight_decay
            self.num_training_steps = self.run_cfg.num_training_steps
            self.lr_warmup_steps = self.run_cfg.lr_warmup_steps
            self.additional_train_logging_period = self.run_cfg.additional_train_logging_period
            self.penetration_weight = self.run_cfg.penetration_weight
        elif self.mode == "eval":
            self.run_cfg = cfg.inference
            # inference-specific params
            self.num_trials = self.run_cfg.num_trials
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        
        # data params
        self.batch_size = self.run_cfg.batch_size
        self.val_batch_size = self.run_cfg.val_batch_size
        # TODO: it is debatable if the module needs to know about the sample size
        self.sample_size = self.run_cfg.sample_size
        self.sample_size_anchor = self.run_cfg.sample_size_anchor

        # diffusion params
        self.diff_steps = self.model_cfg.diff_train_steps
        self.num_wta_trials = self.run_cfg.num_wta_trials
        self.time_based_weighting = self.model_cfg.time_based_weighting

        # TODO: have 3 different noise scale for per-point, translation, and rotation
        self.diff_translation_noise_scale = self.model_cfg.diff_translation_noise_scale 
        self.diff_rotation_noise_scale = self.model_cfg.diff_rotation_noise_scale

        if self.model_cfg.frame_type == "mu":
            self.diffusion = create_diffusion_mu(
                timestep_respacing=None,
                diffusion_steps=self.diff_steps,
                time_based_weighting=self.time_based_weighting,
                rotation_noise_scale=self.diff_rotation_noise_scale,
            )

            print("Initializing TAX3Dv2 Mu-Frame Diffusion Transformer Network")
            print(f"### Translation Noise Scale: {self.diff_translation_noise_scale}")
            print(f"### Rotation Noise Scale: {self.diff_rotation_noise_scale}")
            print("Initializing TAX3Dv2 Mu-Frame Dense Displacement Diffusion Module")
            print(f"### Prediction Reference Frame: {self.pred_frame}")
            print(f"### Reference Noise Scale: {self.noisy_goal_scale}")

        elif self.model_cfg.frame_type == "fixed":
            # TODO: rename this diffusion code to create_diffusion_fixed
            self.diffusion = create_diffusion_ddrd_separate(
                timestep_respacing=None,
                diffusion_steps=self.diff_steps,
                time_based_weighting=self.time_based_weighting,
                rotation_noise_scale=self.diff_rotation_noise_scale,
            )

            print("Initializing TAX3Dv2 Fixed-Frame Diffusion Transformer Network")
            print(f"### Translation Noise Scale: {self.diff_translation_noise_scale}")
            print(f"### Rotation Noise Scale: {self.diff_rotation_noise_scale}")
            print("Initializing TAX3Dv2 Fixed-Frame Dense Displacement Diffusion Module")
            print(f"### Prediction Reference Frame: {self.pred_frame}")
            print(f"### Reference Noise Scale: {self.noisy_goal_scale}")

        else:
            raise ValueError("Choose model_take from [\"tax3dv2_muframe\", \"tax3dv2_fixedframe\"]")
        

    def configure_optimizers(self):
        assert self.mode == "train", "Can only configure optimizers in training mode."
        optimizer = optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.lr_warmup_steps,
            num_training_steps=self.num_training_steps,
        )
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': lr_scheduler, 'interval': 'step'}}

    def get_model_kwargs(self, batch, nun_samples=None):
        """
        Get the model kwargs for the forward pass.
        """
        raise NotImplementedError("This should be implemented in the derived class.")
    
    def update_batch_frames(self, batch, update_labels=False, gmm_model=None):
        """
        Convert data batch from world frame to prediction frame.
        """
        raise NotImplementedError("This should be implemented in the derived class.")

    def get_world_preds(self, batch, num_samples, pc_action, pred_dict):
        """
        Get world frame predictions from the given batch and predictions.
        """
        raise NotImplementedError("This should be implemented in the derived class.")

    def get_viz_args(self, batch, viz_idx):
        """
        Get visualization arguments for wandb logging.
        """
        raise NotImplementedError("This should be implemented in the derived class.")

    def forward(self, batch, t):
        """
        Forward pass to compute diffusion training loss.
        """
        assert "pred_frame" in batch.keys(), "Please run self.update_batch_frames() to update the data batch!"

        ground_truth = batch[self.label_key].permute(0, 2, 1) # channel first
        model_kwargs = self.get_model_kwargs(batch)

        # run diffusion
        # noise = torch.randn_like(ground_truth) * self.noise_scale
        loss_dict = self.diffusion.training_losses(
            model=self.network,
            x_start=ground_truth,
            t=t,
            model_kwargs=model_kwargs,
            object_pc= batch["pene_object_pc"],
            object_normals=batch["pene_normals"],
            T_action2world=batch["T_action2world"],
            penetration_weight = self.penetration_weight,
            # noise=noise,
        )
        loss_r = loss_dict["loss_r"].mean()
        loss_s = loss_dict["loss_s"].mean()
        if "loss_depth" in loss_dict:
            loss_depth = loss_dict["loss_depth"].mean()
        else:
            loss_depth = torch.tensor(0.0, device=self.device)
        
        loss = loss_dict["loss"].mean()

        return None, loss, loss_r, loss_s, loss_depth

    @torch.no_grad()
    def predict(self, batch, num_samples, unflatten=False, progress=True, full_prediction=True):
        """
        Compute prediction for a given batch.

        Args:
            batch: the input batch
            num_samples: the number of samples to generate
            progress: whether to show progress bar
            full_prediction: whether to return full prediction (flow and point, goal and world frame)
        """
        assert "pred_frame" in batch.keys(), "Please run self.update_batch_frames() to update the data batch!"
        pred_frame = batch["pred_frame"].clone()
        pred_frame = expand_pcd(pred_frame, num_samples)

        # TODO: replace bs with batch_size?
        bs, sample_size = batch["pc_action"].shape[:2]
        model_kwargs = self.get_model_kwargs(batch, num_samples)

        # generating latents and running diffusion
        z_s = torch.randn(bs * num_samples, 3, sample_size, device=self.device)

        if self.model_cfg.frame_type == "mu":
            z_r = pred_frame.to(self.device).permute(0,2,1)
        elif self.model_cfg.frame_type == "fixed":
            z_r = torch.randn(bs * num_samples, 3, 1, device=self.device)
        else:
            raise ValueError("Choose model_take from [\"tax3dv2_muframe\", \"tax3dv2_fixedframe\"]")

        # test: in inference, if we trained with translation noise with scale=t, we sample noise from N(0, 1+t)
        #trans_noise_scale = torch.tensor(0.6, device=self.device)
        #z = torch.randn(bs * num_samples, 3, sample_size, device=self.device) * torch.sqrt(1 + trans_noise_scale)

        final_dict, results = self.diffusion.p_sample_loop(
            self.network,
            z_r.shape,
            z_s.shape,
            z_r,
            z_s,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=progress,
            device=self.device,
        )
        pred_r = final_dict["sample_r"]
        pred_s = final_dict["sample_s"]
        pred = pred_s + pred_r        
        results_r = [res["sample_r"] for res in results]
        results_s = [res["sample_s"] for res in results]
        results = [res["sample_r"] + res["sample_s"] for res in results]

        '''
        finetune_frame = pred_r
        z_r = final_dict["prev_sample_r"]
        z_s = pred_s
        final_dict, results = self.diffusion.p_sample_loop_finetune(
            self.network,
            z_r.shape,
            z_s.shape,
            z_r,
            z_s,
            finetune_frame=finetune_frame,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=progress,
            device=self.device,   
        )
        pred_r = final_dict["sample_r"]
        pred_s = final_dict["sample_s"]
        pred = pred_s + pred_r + finetune_frame
        results = [res["sample_r"] + res["sample_s"] + finetune_frame for res in results]
        '''
        pred = pred.permute(0, 2, 1)

        if not full_prediction:
            # only return the prediction type in the goal frame
            return {self.prediction_type: {"pred": pred}}
        else:
            # return full prediction (flow and point, goal and world frame)
            pc_action = model_kwargs["x0"].permute(0, 2, 1)

            # computing flow and point predictions
            if self.prediction_type == "flow":
                '''
                pred_flow = pred
                pred_point = pc_action + pred_flow
                # for flow predictions, convert results to point predictions
                results = [
                    pc_action + res.permute(0, 2, 1) for res in results
                ]'''
                raise NotImplementedError
            elif self.prediction_type == "point":
                pred_point = pred
                pred_flow = pred_point - pc_action
                results = [
                    res.permute(0, 2, 1) for res in results
                ]
                results_r = [
                    res.permute(0, 2, 1) for res in results_r
                ]
                results_s = [
                    res.permute(0, 2, 1) for res in results_s
                ]

            pred_dict = {
                "flow": {
                    "pred": pred_flow,
                },
                "point": {
                    "pred": pred_point,
                },
                "results": results,
                "results_r": results_r,
                "results_s": results_s,
            }

            # compute world frame predictions
            pred_world_dict = self.get_world_preds(batch, num_samples, pc_action, pred_dict)

            pred_dict["flow"]["pred_world"] = pred_world_dict['pred_flow_world']
            pred_dict["point"]["pred_world"] = pred_world_dict['pred_point_world']
            pred_dict["pred_frame_world"] = pred_world_dict['pred_frame_world']
            pred_dict["init_action_world"] = pred_world_dict['pc_action_world']
            pred_dict["results_world"] = pred_world_dict['results_world']
            pred_dict["results_r_world"] = pred_world_dict['results_r_world']
            pred_dict["results_s_world"] = pred_world_dict['results_s_world']

            # if the material is rigid, we also output estimated translation and rotation
            if self.dataset_cfg.material == "rigid":
                scaling_factor = self.dataset_cfg.pcd_scale_factor

                action_world_scaled = pred_world_dict['pc_action_world'] / scaling_factor
                pred_world_scaled = pred_world_dict['pred_point_world'] / scaling_factor

                T = svd_estimation(source=action_world_scaled, target=pred_world_scaled, return_magnitude=False)
                pred_dict["pred_T"] = T

            return pred_dict

    def predict_wta(self, batch, num_samples):
        """
        Predict WTA (winner-take-all) samples, and compute WTA metrics. Unlike predict, this 
        function assumes the ground truth is available.

        Args:
            batch: the input batch
            num_samples: the number of samples to generate
        """
        assert "pred_frame" in batch.keys(), "Please run self.update_batch_frames to update the data batch!"

        seg = batch["seg"].to(self.device)
        ground_truth_point_world = batch["pc_world"].to(self.device)
        scaling_factor = self.dataset_cfg.pcd_scale_factor

        # re-shaping and expanding for winner-take-all
        bs = ground_truth_point_world.shape[0]
        seg = expand_pcd(seg, num_samples)
        ground_truth_point_world = expand_pcd(ground_truth_point_world, num_samples)

        # generating diffusion predictions
        pred_dict = self.predict(
            batch, num_samples, unflatten=False, progress=True
        )
        # pred = pred_dict[self.prediction_type]["pred"]
        pred_point_world = pred_dict["point"]["pred_world"]

        # TODO: this should happen inside get model kwargs
        pred_point_world_scaled = pred_point_world / scaling_factor
        ground_truth_point_world_scaled = ground_truth_point_world / scaling_factor

        # computing error metrics
        seg = seg == 0

        rmse = flow_rmse(pred_point_world_scaled, ground_truth_point_world_scaled, mask=True, seg=seg).reshape(bs, num_samples)
        pred_point_world = pred_point_world.reshape(bs, num_samples, -1, 3)

        # computing winner-take-all metrics
        winner = torch.argmin(rmse, dim=-1)
        rmse_wta = rmse[torch.arange(bs), winner]
        pred_point_world_wta = pred_point_world[torch.arange(bs), winner]

        if self.dataset_cfg.material == "rigid":
            translation_errs, rotation_errs = svd_estimation(source=pred_point_world_scaled.reshape(bs * num_samples, -1, 3), target=ground_truth_point_world_scaled, return_magnitude=True)

            translation_errs = translation_errs.reshape(bs, num_samples)
            rotation_errs = rotation_errs.reshape(bs, num_samples)

            trans_winner = torch.argmin(translation_errs, dim=-1)
            rot_winner = torch.argmin(rotation_errs, dim=-1)

            translation_err_wta = translation_errs[torch.arange(bs), trans_winner]
            rotation_err_wta = rotation_errs[torch.arange(bs), rot_winner]
            
            return {
                "pred_point_world": pred_point_world,
                "pred_point_world_wta": pred_point_world_wta,
                "rmse": rmse,
                "rmse_wta": rmse_wta,
                "trans": translation_errs,
                "trans_wta": translation_err_wta,
                "rot": rotation_errs,
                "rot_wta": rotation_err_wta,
            }
        
        else:
            return {
                "pred_point_world": pred_point_world,
                "pred_point_world_wta": pred_point_world_wta,
                "rmse": rmse,
                "rmse_wta": rmse_wta,
            }

    def log_viz_to_wandb(self, batch, pred_wta_dict, tag):
        """
        Log visualizations to wandb.

        Args:
            batch: the input batch
            pred_wta_dict: the prediction dictionary
            tag: the tag to use for logging
        """
        # pick a random sample in the batch to visualize
        viz_idx = np.random.randint(0, batch["pc"].shape[0])
        pred_action_viz = pred_wta_dict["pred_point_world"][viz_idx, 0, :, :3]
        pred_action_wta_viz = pred_wta_dict["pred_point_world_wta"][viz_idx, :, :3]
        viz_args = self.get_viz_args(batch, viz_idx)

        # logging predicted vs ground truth point cloud
        viz_args["pred_action_viz"] = pred_action_viz
        predicted_vs_gt = viz_predicted_vs_gt(**viz_args)
        wandb.log({f"{tag}/predicted_vs_gt": predicted_vs_gt})

        # logging predicted vs ground truth point cloud (wta)
        viz_args["pred_action_viz"] = pred_action_wta_viz
        predicted_vs_gt_wta = viz_predicted_vs_gt(**viz_args)
        wandb.log({f"{tag}/predicted_vs_gt_wta": predicted_vs_gt_wta})

    def training_step(self, batch):
        """
        Training step for the module. Logs training metrics and visualizations to wandb.
        """
        self.train()
        bs = batch[self.label_key].shape[0]
        t = torch.randint(
            0, self.diff_steps, (bs,), device=self.device
        ).long()

        batch = self.update_batch_frames(batch, update_labels=True)

        _, loss, loss_r, loss_s, loss_depth = self(batch, t)
        #########################################################
        # logging training metrics
        #########################################################
        self.log_dict(
            {"train/loss": loss,
            "train/loss_r": loss_r,
            "train/loss_s": loss_s,
            "train/loss_depth": loss_depth,},
            add_dataloader_idx=False,
            prog_bar=True,
        )

        # determine if additional logging should be done
        do_additional_logging = (
            self.global_step % self.additional_train_logging_period == 0
        )

        '''
        # Perform gradient monitoring after backward pass
        max_grad = 0
        for name, param in self.network.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()  # Compute L2 norm of gradient
                max_grad = max(max_grad, grad_norm)
        
        # Log gradient norm
        self.log("train/max_grad_norm", max_grad, prog_bar=True)
        if max_grad > 1e3:  # Example threshold for gradient explosion
            self.log("train/grad_warning", 1.0)
            print(f"Warning: Exploding gradient detected! Max Gradient Norm: {max_grad:.4f}")
        '''

        # additional logging
        if do_additional_logging:
            self.eval()
            with torch.no_grad():
                # winner-take-all predictions
                train_dataloader = self.trainer.datamodule.train_dataloader()
                train_dataloader.dataset.set_eval_mode(True)

                batch = next(iter(train_dataloader))  # Fetch new batch with eval_mode=True
                batch = self.update_batch_frames(batch, update_labels=True)

                # TODO: Debug why without this line, it will rasie gpu/cpu different device error
                batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                pred_wta_dict = self.predict_wta(batch, self.num_wta_trials)

                ####################################################
                # logging training wta metrics
                ####################################################
                self.log_dict(
                    {
                        "train/rmse": pred_wta_dict["rmse"].mean(),
                        "train/rmse_wta": pred_wta_dict["rmse_wta"].mean(),
                    },
                    add_dataloader_idx=False,
                    prog_bar=True,
                )

                if self.dataset_cfg.material == "rigid":
                    ####################################################
                    # logging rigid metrics (translation & rotation)
                    ####################################################
                    self.log_dict(
                        {
                            "train/trans": pred_wta_dict["trans"].mean(),
                            "train/trans_wta": pred_wta_dict["trans_wta"].mean(),
                            "train/rot": pred_wta_dict["rot"].mean(),
                            "train/rot_wta": pred_wta_dict["rot_wta"].mean(),
                        },
                        add_dataloader_idx=False,
                        prog_bar=True,
                    )

                ####################################################
                # logging visualizations
                ####################################################
                self.log_viz_to_wandb(batch, pred_wta_dict, "train")

                train_dataloader.dataset.set_eval_mode(False)


        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Validation step for the module. Logs validation metrics and visualizations to wandb.
        """
        self.eval()
        with torch.no_grad():
            # winner-take-all predictions
            batch = self.update_batch_frames(batch, update_labels=True)
            pred_wta_dict = self.predict_wta(batch, self.num_wta_trials)
        
        ####################################################
        # logging validation wta metrics
        ####################################################
        self.log_dict(
            {
                f"val_rmse_{dataloader_idx}": pred_wta_dict["rmse"].mean(),
                f"val_rmse_wta_{dataloader_idx}": pred_wta_dict["rmse_wta"].mean(),
            },
            add_dataloader_idx=False,
            prog_bar=True,
        )

        if self.dataset_cfg.material == "rigid":
            ####################################################
            # logging rigid metrics (translation & rotation)
            ####################################################
            self.log_dict(
                {
                    f"val_trans_{dataloader_idx}": pred_wta_dict["trans"].mean(),
                    f"val_trans_wta_{dataloader_idx}": pred_wta_dict["trans_wta"].mean(),
                    f"val_rot_{dataloader_idx}": pred_wta_dict["rot"].mean(),
                    f"val_rot_wta_{dataloader_idx}": pred_wta_dict["rot_wta"].mean(),
                },
                add_dataloader_idx=False,
                prog_bar=True,
            )

        ####################################################
        # logging visualizations
        ####################################################
        self.log_viz_to_wandb(batch, pred_wta_dict, f"val_{dataloader_idx}")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Prediction step for model evaluation. Computes winner-take-all metrics.
        """
        if self.dataset_cfg.material == "rigid":
            # winner-take-all predictions
            batch = self.update_batch_frames(batch, update_labels=True)
            pred_wta_dict = self.predict_wta(batch, self.num_wta_trials)
            return {
                "rmse": pred_wta_dict["rmse"],
                "rmse_wta": pred_wta_dict["rmse_wta"],
                "trans": pred_wta_dict["trans"].mean(),
                "trans_wta": pred_wta_dict["trans_wta"].mean(),
                "rot": pred_wta_dict["rot"].mean(),
                "rot_wta": pred_wta_dict["rot_wta"].mean(),
            }
        else:
            # winner-take-all predictions
            batch = self.update_batch_frames(batch, update_labels=True)
            pred_wta_dict = self.predict_wta(batch, self.num_wta_trials)
            return {
                "rmse": pred_wta_dict["rmse"],
                "rmse_wta": pred_wta_dict["rmse_wta"],
            }
    

class TAX3Dv2MuFrameModule(TAX3Dv2BaseModule):
    """
    Object-centric Diffusion module. Applies cross attention between action and anchor objects.
    """
    def __init__(self, network, cfg) -> None:
        super().__init__(network, cfg)

    def get_model_kwargs(self, batch, num_samples=None):
        pc_action = batch["pc_action"].to(self.device)
        pc_anchor = batch["pc_anchor"].to(self.device)

        if num_samples is not None:
            # expand point clouds if num_samples is provided; used for WTA predictions
            pc_action = expand_pcd(pc_action, num_samples)
            pc_anchor = expand_pcd(pc_anchor, num_samples)
        
        pc_action = pc_action.permute(0, 2, 1) # channel first
        pc_anchor = pc_anchor.permute(0, 2, 1) # channel first
        model_kwargs = dict(x0=pc_action, y=pc_anchor)

        # Extract relative position, if necessary.
        if self.model_cfg.rel_pos:
            rel_pos = batch["rel_pos"].to(self.device)
            if num_samples is not None:
                rel_pos = expand_pcd(rel_pos, num_samples)
            model_kwargs["rel_pos"] = rel_pos

        return model_kwargs
    
    def get_world_preds(self, batch, num_samples, pc_action, pred_dict):
        """
        Get world-frame predictions from the given batch and predictions.
        """
        T_action2world = Transform3d(
            matrix=expand_pcd(batch["T_action2world"].to(self.device), num_samples)
        )
        T_goal2world = Transform3d(
            matrix=expand_pcd(batch["T_goal2world"].to(self.device), num_samples)
        )

        # pred_frame = expand_pcd(batch["pred_frame"].to(self.device), num_samples)
        action_context_frame = expand_pcd(batch["action_context_frame"].to(self.device), num_samples)

        pred_point_world = T_goal2world.transform_points(pred_dict["point"]["pred"])
        pc_action_world = T_action2world.transform_points(pc_action + action_context_frame)
        pred_flow_world = pred_point_world - pc_action_world
        results_world = [
            T_goal2world.transform_points(res) for res in pred_dict["results"]
        ]
        return pred_flow_world, pred_point_world, pc_action_world, results_world

    def update_batch_frames(self, batch, update_labels=False, gmm_model=None):
        # Using GMM model, if provided.
        if gmm_model is not None:
            assert self.model_cfg.pred_frame == "noisy_goal", "GMM model can only be used with noisy goal prediction frame!"
            gmm_pred = gmm_model(batch)
            gmm_probs, gmm_means, gmm_anchor_frame = gmm_pred["probs"], gmm_pred["means"], gmm_pred["anchor_frame"]
            idxs = torch.multinomial(gmm_probs.squeeze(-1), 1).squeeze()
            sampled_noisy_goals = (
                gmm_means[torch.arange(gmm_means.shape[0]), idxs] + gmm_anchor_frame.squeeze(1)
            ).cpu()
            batch["noisy_goal"] = sampled_noisy_goals

            # Also add gmm predictions to batch for visualization.
            batch["gmm_probs"] = gmm_probs
            batch["gmm_means"] = gmm_means
            batch["sampled_idxs"] = idxs

        # Processing prediction frame.
        if self.model_cfg.pred_frame == "anchor_center":
            raise NotImplementedError("Mu-frame not implemented for anchor centroid prediction!")
        elif self.model_cfg.pred_frame == "noisy_goal":
            pred_frame = batch["noisy_goal"].unsqueeze(1)
        else:
            raise ValueError(f"Invalid prediction reference frame: {self.pred_frame}")

        # Processing action context frame.
        if self.model_cfg.action_context_frame == "action_center":
            action_context_frame = batch["pc_action"].mean(axis=1, keepdim=True)
        else:
            raise ValueError(f"Invalid action context frame: {self.model_cfg.action_context_frame}")

        # Update scene-as-anchor, if necessary.
        if self.model_cfg.scene_anchor:
            batch["pc_anchor"] = torch.cat(
                [batch["pc_anchor"], batch["pc_action"]], dim=1
            )
       
        batch["pc_action"] = batch["pc_action"] - action_context_frame

        # Updating labels, if necessary.
        if update_labels:
            # Compute ground truth point cloud in world frame.
            T_goal2world = Transform3d(
                matrix=batch["T_goal2world"]#.to(self.device)
            )
            batch["pc_world"] = T_goal2world.transform_points(batch["pc"])

            # Put flow labels in prediction frame.
            batch["flow"] = batch["flow"] + action_context_frame
        
        # Compute relative position, if necessary.
        if self.model_cfg.rel_pos:
            batch["rel_pos"] = action_context_frame - pred_frame

        batch["pred_frame"] = pred_frame
        batch["action_context_frame"] = action_context_frame
        return batch

    def get_viz_args(self, batch, viz_idx):
        """
        Get visualization arguments for wandb logging.
        """
        assert "pred_frame" in batch.keys(), "Please run self.update_batch_frames() to update the data batch!"

        pc_pos_viz = batch["pc_world"][viz_idx, :, :3]
        pc_action_viz = batch["pc_action"][viz_idx, :, :3]
        pc_anchor_viz = batch["pc_anchor"][viz_idx, :, :3]

        # Put action and anchor point clouds in the world frame.
        # pred_frame = batch["pred_frame"][viz_idx, :, :3]
        action_context_frame = batch["action_context_frame"][viz_idx, :, :3]

        T_action2world = Transform3d(
            matrix=batch["T_action2world"][viz_idx].to(self.device)
        )
        T_goal2world = Transform3d(
            matrix=batch["T_goal2world"][viz_idx].to(self.device)
        )

        pc_action_viz = T_action2world.transform_points(pc_action_viz + action_context_frame)
        pc_anchor_viz = T_goal2world.transform_points(pc_anchor_viz)

        viz_args = {
            "pc_pos_viz": pc_pos_viz,
            "pc_action_viz": pc_action_viz,
            "pc_anchor_viz": pc_anchor_viz,
        }
        return viz_args


class TAX3Dv2FixedFrameModule(TAX3Dv2BaseModule):
    """
    Object-centric Diffusion module. Applies cross attention between action and anchor objects.
    """
    def __init__(self, network, cfg) -> None:
        super().__init__(network, cfg)

    def get_model_kwargs(self, batch, num_samples=None):
        pc_action = batch["pc_action"].to(self.device)
        pc_anchor = batch["pc_anchor"].to(self.device)
        if num_samples is not None:
            # expand point clouds if num_samples is provided; used for WTA predictions
            pc_action = expand_pcd(pc_action, num_samples)
            pc_anchor = expand_pcd(pc_anchor, num_samples)
        
        pc_action = pc_action.permute(0, 2, 1) # channel first
        pc_anchor = pc_anchor.permute(0, 2, 1) # channel first
        model_kwargs = dict(x0=pc_action, y=pc_anchor)

        # Extract relative position, if necessary.
        if self.model_cfg.rel_pos:
            rel_pos = batch["rel_pos"].to(self.device)
            if num_samples is not None:
                rel_pos = expand_pcd(rel_pos, num_samples)
            model_kwargs["rel_pos"] = rel_pos
            
        return model_kwargs
    
    def get_world_preds(self, batch, num_samples, pc_action, pred_dict):
        """
        Get world-frame predictions from the given batch and predictions.
        """
        T_action2world = Transform3d(
            matrix=expand_pcd(batch["T_action2world"].to(self.device), num_samples)
        )
        T_goal2world = Transform3d(
            matrix=expand_pcd(batch["T_goal2world"].to(self.device), num_samples)
        )

        pred_frame = expand_pcd(batch["pred_frame"].to(self.device), num_samples)
        action_context_frame = expand_pcd(batch["action_context_frame"].to(self.device), num_samples)

        # pred_point_world = T_goal2world.transform_points(pred_dict["point"]["pred"] + pred_frame)
        # pc_action_world = T_action2world.transform_points(pc_action + action_context_frame)

        # Scale point clouds, if necessary.
        if self.object_scale is not None or self.scene_scale is not None:
            scale = expand_pcd(batch["pc_scale"].to(self.device), num_samples)
            pred_frame = pred_frame * scale
            action_context_frame = action_context_frame * scale
        else:
            scale = 1.0
            
        pred_frame_world = T_goal2world.transform_points(pred_frame)
        pred_point_world = T_goal2world.transform_points(pred_dict["point"]["pred"]*scale + pred_frame)
        pc_action_world = T_action2world.transform_points(pc_action*scale + action_context_frame)

        pred_flow_world = pred_point_world - pc_action_world
        results_world = [
            T_goal2world.transform_points(res*scale + pred_frame) for res in pred_dict["results"]
        ]
        results_r_world = [
            T_goal2world.transform_points(res + pred_frame) for res in pred_dict["results_r"]
        ]
        results_s_world = [
            T_goal2world.transform_points(res + pred_frame) for res in pred_dict["results_s"]
        ]

        pred_world_dict = {
            "pred_flow_world": pred_flow_world,
            "pred_point_world": pred_point_world,
            "pc_action_world": pc_action_world,
            "pred_frame_world": pred_frame_world,
            "results_world": results_world,
            "results_r_world": results_r_world,
            "results_s_world": results_s_world,
        }

        return pred_world_dict

    def update_batch_frames(self, batch, update_labels=False, gmm_model=None):
        # Using GMM model, if provided.
        if gmm_model is not None:
            assert self.model_cfg.pred_frame == "noisy_goal", "GMM model can only be used with noisy goal prediction frame!"
            gmm_pred = gmm_model(batch)
            gmm_probs, gmm_means, gmm_anchor_frame = gmm_pred["probs"], gmm_pred["means"], gmm_pred["anchor_frame"]
            idxs = torch.multinomial(gmm_probs.squeeze(-1), 1).squeeze()
            sampled_noisy_goals = (
                gmm_means[torch.arange(gmm_means.shape[0]), idxs] + gmm_anchor_frame.squeeze(1)
            ).cpu()
            batch["noisy_goal"] = sampled_noisy_goals

            # Also add gmm predictions to batch for visualization.
            batch["gmm_probs"] = gmm_probs
            batch["gmm_means"] = gmm_means
            batch["sampled_idxs"] = idxs

        # Processing prediction frame.
        if self.model_cfg.pred_frame == "anchor_center":
            pred_frame = batch["pc_anchor"].mean(axis=1, keepdim=True)
        elif self.model_cfg.pred_frame == "noisy_goal":
            pred_frame = batch["noisy_goal"].unsqueeze(1)
        else:
            raise ValueError(f"Invalid prediction frame: {self.model_cfg.pred_frame}")

        # Processing action context frame.
        if self.model_cfg.action_context_frame == "action_center":
            action_context_frame = batch["pc_action"].mean(axis=1, keepdim=True)
        else:
            raise ValueError(f"Invalid action context frame: {self.model_cfg.action_context_frame}")
        
        # Re-scale point cloud inputs, if necessary.
        if self.object_scale is not None or self.scene_scale is not None:
            # Computing scale factor.
            scale = self.object_scale if self.object_scale is not None else self.scene_scale
            points = batch["pc_action"] if self.object_scale is not None else torch.cat([batch["pc_action"], batch["pc_anchor"]], dim=1)
            point_dists = points - points.mean(axis=1, keepdim=True)
            # point_scale = torch.linalg.norm(point_dists, dim=2, keepdim=True).mean(dim=1, keepdim=True)
            point_scale = torch.linalg.norm(point_dists, dim=2, keepdim=True).max(dim=1, keepdim=True).values

            # Updating point clouds.
            batch["pc_action"] = batch["pc_action"] * scale / point_scale
            batch["pc_anchor"] = batch["pc_anchor"] * scale / point_scale
            pred_frame = pred_frame * scale / point_scale
            action_context_frame = action_context_frame * scale / point_scale
            batch["pc_scale"] = point_scale / scale

        # Update scene-as-anchor, if necessary.
        if self.model_cfg.scene_anchor:
            batch["pc_anchor"] = torch.cat(
                [batch["pc_anchor"], batch["pc_action"]], dim=1
            )

        # Put object and scene point clouds in the corresponding frames.
        batch["pc_anchor"] = batch["pc_anchor"] - pred_frame
        batch["pc_action"] = batch["pc_action"] - action_context_frame

        # Updating labels, if necessary.
        if update_labels:
            # Compute ground truth point cloud in world frame.
            T_goal2world = Transform3d(
                matrix=batch["T_goal2world"]#.to(self.device)
            )
            batch["pc_world"] = T_goal2world.transform_points(batch["pc"])
            
            # Scale ground truth point cloud, if necessary.
            if self.object_scale is not None or self.scene_scale is not None:
                batch[self.label_key] = batch[self.label_key] * scale / point_scale

            # Put point and flow labels in prediction frame.
            # TODO: the flow computation is technically bugged here, should also be scaled
            batch["pc"] = batch["pc"] - pred_frame
            batch["flow"] = batch["flow"] - pred_frame + action_context_frame
        
        # Compute relative position, if necessary.
        if self.model_cfg.rel_pos:
            batch["rel_pos"] = action_context_frame - pred_frame
        
        batch["pred_frame"] = pred_frame
        batch["action_context_frame"] = action_context_frame
        return batch

    def get_viz_args(self, batch, viz_idx):
        """
        Get visualization arguments for wandb logging.
        """
        assert "pred_frame" in batch.keys(), "Please run self.update_batch_frames() to update the data batch!"

        pc_pos_viz = batch["pc_world"][viz_idx, :, :3]
        pc_action_viz = batch["pc_action"][viz_idx, :, :3]
        pc_anchor_viz = batch["pc_anchor"][viz_idx, :, :3]

        # Put action and anchor point clouds in the world frame.
        pred_frame = batch["pred_frame"][viz_idx, :, :3]
        action_context_frame = batch["action_context_frame"][viz_idx, :, :3]

        T_action2world = Transform3d(
            matrix=batch["T_action2world"][viz_idx].to(self.device)
        )
        T_goal2world = Transform3d(
            matrix=batch["T_goal2world"][viz_idx].to(self.device)
        )

        # Scale point clouds, if necessary.
        if self.object_scale is not None or self.scene_scale is not None:
            scale = batch["pc_scale"][viz_idx]
            pc_action_viz = pc_action_viz * scale
            pc_anchor_viz = pc_anchor_viz * scale
            pred_frame = pred_frame * scale
            action_context_frame = action_context_frame * scale

        pc_action_viz = T_action2world.transform_points(pc_action_viz + action_context_frame)
        pc_anchor_viz = T_goal2world.transform_points(pc_anchor_viz + pred_frame)

        viz_args = {
            "pc_pos_viz": pc_pos_viz,
            "pc_action_viz": pc_action_viz,
            "pc_anchor_viz": pc_anchor_viz,
        }
        return viz_args