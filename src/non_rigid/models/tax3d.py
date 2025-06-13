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
from non_rigid.models.dit.diffusion import create_diffusion
from non_rigid.models.dit.models import (
    DiT_PointCloud_Cross
)
from non_rigid.utils.logging_utils import viz_predicted_vs_gt
from non_rigid.utils.pointcloud_utils import expand_pcd

def DiT_PointCloud_Cross_xS(**kwargs):
    return DiT_PointCloud_Cross(depth=5, hidden_size=128, num_heads=4, **kwargs)

DiT_models = {
    "DiT_PointCloud_Cross_xS": DiT_PointCloud_Cross_xS
}


def get_model(model_cfg):
    model_name = f"DiT_PointCloud_Cross_{model_cfg.size}"
    return DiT_models[model_name]


class DiffusionTransformerNetwork(nn.Module):
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
    
    def forward(self, x, t, **kwargs):
        return self.dit(x, t, **kwargs)
    


class DenseDisplacementDiffusionModule(L.LightningModule):
    """
    Generalized Dense Displacement Diffusion (DDD) module that handles model training, inference, 
    evaluation, and visualization. This module is inherited and overriden by scene-level and 
    object-centric modules.
    """
    def __init__(self, network, cfg) -> None:
        super().__init__()
        self.network = network
        self.model_cfg = cfg.model
        self.dataset_cfg = cfg.dataset
        self.wandb_cfg = cfg.wandb
        self.prediction_type = self.model_cfg.type # flow or point
        self.mode = cfg.mode # train or eval
        self.pred_frame = self.model_cfg.pred_frame
        self.noisy_goal_scale = self.model_cfg.noisy_goal_scale

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

        # mode-specific processing
        if self.mode == "train":
            self.run_cfg = cfg.training
            # training-specific params
            self.lr = self.run_cfg.lr
            self.weight_decay = self.run_cfg.weight_decay
            self.num_training_steps = self.run_cfg.num_training_steps
            self.lr_warmup_steps = self.run_cfg.lr_warmup_steps
            self.additional_train_logging_period = self.run_cfg.additional_train_logging_period
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
        # self.noise_schedule = model_cfg.diff_noise_schedule
        # self.noise_scale = model_cfg.diff_noise_scale
        self.diff_steps = self.model_cfg.diff_train_steps # TODO: rename to diff_steps
        self.num_wta_trials = self.run_cfg.num_wta_trials
        self.diff_translation_noise_scale = self.model_cfg.diff_translation_noise_scale
        self.diff_rotation_noise_scale = self.model_cfg.diff_rotation_noise_scale
        self.diffusion = create_diffusion(
            timestep_respacing=None,
            diffusion_steps=self.diff_steps,
            # noise_schedule=self.noise_schedule,
        )

        # TODO: make this a info helper method
        print("Initializing TAX3D Diffusion Transformer Network")
        print(f"### Translation Noise Scale: {self.diff_translation_noise_scale}")
        print(f"### Rotation Noise Scale: {self.diff_rotation_noise_scale}")
        print("Initializing TAX3D Dense Displacement Diffusion Module")
        print(f"### Prediction Frame: {self.pred_frame}")
        print(f"### Prediction Frame Noise Scale: {self.noisy_goal_scale}")

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
    
    def get_world_preds(self, batch, num_samples, pc_action, pred_dict):
        """
        Get world frame predictions from the given batch and predictions.
        """
        raise NotImplementedError("This should be implemented in the derived class.")
    
    def update_batch_frames(self, batch, update_labels=False, gmm_model=None):
        """
        Convert data batch from world frame to prediction frame.
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
            translation_noise_scale=self.diff_translation_noise_scale,
            rotation_noise_scale=self.diff_rotation_noise_scale,
            model_kwargs=model_kwargs,
            # noise=noise,
        )
        loss = loss_dict["loss"].mean()
        return None, loss

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
        
        bs, sample_size = batch["pc_action"].shape[:2]
        model_kwargs = self.get_model_kwargs(batch, num_samples)

        # generating latents and running diffusion
        z = torch.randn(bs * num_samples, 3, sample_size, device=self.device)
        '''
        # test: in inference, if we trained with rotation noise we ought to also sample from it
        if self.model_cfg.diff_rotation_noise_scale:
            pc = batch["pc_action"].permute(0,2,1)
            pc = expand_pcd(pc, num_samples)
            N, _, _ = z.shape  # e.g. [batch_size, 3, num_points]

            # 1. Sample random rotation axes and normalize them.
            random_axis = torch.randn(N, 3, device=pc.device)
            random_axis = random_axis / random_axis.norm(dim=1, keepdim=True)

            # 2. Sample random rotation angles (in degrees) and convert to radians.
            random_angle = torch.randn(N, device=pc.device) * self.model_cfg.diff_rotation_noise_scale
            random_angle_rad = random_angle * (np.pi / 180)

            # 3. Compute sin and cos for each angle.
            sin_theta = torch.sin(random_angle_rad)
            cos_theta = torch.cos(random_angle_rad)
            one_minus_cos = 1 - cos_theta

            # 4. Extract axis components.
            x_axis = random_axis[:, 0]
            y_axis = random_axis[:, 1]
            z_axis = random_axis[:, 2]
            zeros = torch.zeros_like(x_axis)

            # 5. Construct the skew-symmetric cross-product matrices for each axis.
            #    Each K is of shape (3,3), and K will have shape (N, 3, 3)
            K = torch.stack([torch.stack([zeros, -z_axis, y_axis], dim=1),
                        torch.stack([z_axis, zeros, -x_axis], dim=1),
                        torch.stack([-y_axis, x_axis, zeros], dim=1)], dim=1)

            # 6. Compute K squared (batched matrix multiplication)
            K2 = torch.bmm(K, K)

            # 7. Create the identity matrix for each batch element.
            I = torch.eye(3, device=pc.device).unsqueeze(0).repeat(N, 1, 1)

            # 8. Compute the rotation matrices using the Rodrigues formula:
            #    R = I + sin(theta)*K + (1-cos(theta))*(K^2)
            R = I + sin_theta.view(N, 1, 1) * K + one_minus_cos.view(N, 1, 1) * K2

            # 9. Apply the rotation matrices to the point clouds.
            #    x_start: [N, 3, P] -> rotated_pc: [N, 3, P]
            rotated_pc = torch.bmm(R, pc)

            # 10. The rotation noise is the difference between the rotated and original points.
            rotation_noise = rotated_pc - pc
            z = z + rotation_noise
        '''
        # test: in inference, if we trained with translation noise with scale=t, we sample noise from N(0, 1+t)
        #trans_noise_scale = torch.tensor(0.6, device=self.device)
        #z = torch.randn(bs * num_samples, 3, sample_size, device=self.device) * torch.sqrt(1 + trans_noise_scale)

        pred, results, _ = self.diffusion.p_sample_loop(
            self.network,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=progress,
            device=self.device,
        )
        pred = pred.permute(0, 2, 1)

        if not full_prediction:
            # only return the prediction type in the goal frame
            return {self.prediction_type: {"pred": pred}}
        else:
            # return full prediction (flow and point, goal and world frame)
            pc_action = model_kwargs["x0"].permute(0, 2, 1)

            # computing flow and point predictions
            if self.prediction_type == "flow":
                pred_flow = pred
                pred_point = pc_action + pred_flow
                # for flow predictions, convert results to point predictions
                results = [
                    pc_action + res.permute(0, 2, 1) for res in results
                ]
            elif self.prediction_type == "point":
                pred_point = pred
                pred_flow = pred_point - pc_action
                results = [
                    res.permute(0, 2, 1) for res in results
                ]

            pred_dict = {
                "flow": {
                    "pred": pred_flow,
                },
                "point": {
                    "pred": pred_point,
                },
                "results": results,
            }

            # compute world frame predictions
            pred_flow_world, pred_point_world, pc_action_world, results_world = self.get_world_preds(
                batch, num_samples, pc_action, pred_dict
            )
            pred_dict["flow"]["pred_world"] = pred_flow_world
            pred_dict["point"]["pred_world"] = pred_point_world
            pred_dict["results_world"] = results_world

            # if the material is rigid, we also output estimated translation and rotation
            if self.dataset_cfg.material == "rigid":
                scaling_factor = self.dataset_cfg.pcd_scale_factor

                action_world_scaled = pc_action_world / scaling_factor
                pred_world_scaled = pred_point_world / scaling_factor

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
        assert "pred_frame" in batch.keys(), "Please run self.update_batch_frames() to update the data batch!"
        
        seg = batch["seg"].to(self.device)
        ground_truth_point_world = batch["pc_world"].to(self.device)
        scaling_factor = self.dataset_cfg.pcd_scale_factor

        # re-shaping and expanding for winner-take-all
        bs = ground_truth_point_world.shape[0]
        seg = expand_pcd(seg, num_samples)
        ground_truth_point_world = expand_pcd(ground_truth_point_world, num_samples)

        # generating diffusion predictions
        pred_dict = self.predict(
            batch, num_samples, unflatten=False, progress=True, full_prediction=True,
        )
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
        _, loss = self(batch, t)
        #########################################################
        # logging training metrics
        #########################################################
        self.log_dict(
            {"train/loss": loss},
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
            # TODO: VERIFY WHETHER THIS SHOULD BE TURNED INTO EVAL & NO GRAD!
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
            pred_wta_dict = self.predict_wta(batch, self.num_wta_trials)
            return {
                "rmse": pred_wta_dict["rmse"],
                "rmse_wta": pred_wta_dict["rmse_wta"],
            }
    
class CrossDisplacementModule(DenseDisplacementDiffusionModule):
    """
    Object-centric DDD module. Applies cross attention between action and anchor objects.
    """
    def __init__(self, network, cfg) -> None:
        super().__init__(network, cfg)

    def get_model_kwargs(self, batch, num_samples=None):
        pc_action = batch["pc_action"].to(self.device)
        pc_anchor = batch["pc_anchor"].to(self.device)

        # Expand point clouds, if necessary; used for WTA predictions.
        if num_samples is not None:
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

        pred_point_world = T_goal2world.transform_points(pred_dict["point"]["pred"] + pred_frame)
        pc_action_world = T_action2world.transform_points(pc_action + action_context_frame)
        pred_flow_world = pred_point_world - pc_action_world
        results_world = [
            T_goal2world.transform_points(res + pred_frame) for res in pred_dict["results"]
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
        
        # Update scene-as-anchor, if necessary.
        if self.model_cfg.scene_anchor:
            batch["pc_anchor"] = torch.cat(
                [batch["pc_anchor"], batch["pc_action"]], dim=1
            )
            
        batch["pc_anchor"] = batch["pc_anchor"] - pred_frame
        batch["pc_action"] = batch["pc_action"] - action_context_frame

        # Updating labels, if necessary.
        if update_labels:
            # Compute ground truth point cloud in world frame.
            T_goal2world = Transform3d(
                matrix=batch["T_goal2world"]#.to(self.device)
            )
            batch["pc_world"] = T_goal2world.transform_points(batch["pc"])

            # Put point and flow labels in prediction frame.
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
        Get visualization arguments for wandb logging. Point clouds are visualized in the world 
        frame for consistency.
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

        pc_action_viz = T_action2world.transform_points(pc_action_viz + action_context_frame)
        pc_anchor_viz = T_goal2world.transform_points(pc_anchor_viz + pred_frame)

        viz_args = {
            "pc_pos_viz": pc_pos_viz,
            "pc_action_viz": pc_action_viz,
            "pc_anchor_viz": pc_anchor_viz,
        }
        return viz_args