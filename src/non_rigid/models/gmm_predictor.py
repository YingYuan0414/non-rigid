import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from plotly import graph_objects as go
from plotly.subplots import make_subplots

import torch_geometric.data as tgd

from non_rigid.models.dit.models import DiT_PointCloud_Cross


#################################################################################
#                               GMM Predictor Model                             #
#################################################################################

class FrameGMMPredictor(nn.Module):
    def __init__(self, network, model_cfg, device):
        super(FrameGMMPredictor, self).__init__()

        # Hacky architecture assertions to enforce model configuration.
        assert model_cfg.name == "df_cross", "FrameGMMPredictor is only compatible with df_cross model."
        assert model_cfg.in_channels == 3, "FrameGMMPredictor is only compatible with 3 input channels."
        assert model_cfg.learn_sigma == True, "FrameGMMPredictor must learn sigma."
        # assert model_cfg.point_encoder == "mlp", "FrameGMMPredictor is only compatible with MLP point encoder."
        assert model_cfg.joint_encode == False, "FrameGMMPredictor is not compatible with joint encoding."
        assert model_cfg.feature == False, "FrameGMMPredictor is not compatible with flow/recon features."
        assert model_cfg.rel_pos == True, "FrameGMMPredictor must use relative position embedding."

        # Hacky data-processing assertions to enforce model configuration.
        assert model_cfg.pred_frame == "anchor_center"
        assert model_cfg.noisy_goal_scale == 0.0
        assert model_cfg.action_context_frame == "action_center"

        self.model_cfg = model_cfg
        self.device = device
        self.network = network
        self.network.to(self.device)
    
    def forward(self, batch):
        pc_action = batch["pc_action"].to(self.device).permute(0, 2, 1).float()
        pc_anchor = batch["pc_anchor"].to(self.device).permute(0, 2, 1).float()

        # TODO: handle point cloud scaling.
        if self.model_cfg.object_scale is not None or self.model_cfg.scene_scale is not None:
            # Computing scale factor.
            scale = self.model_cfg.pcd_scale

            action_dists = pc_action - pc_action.mean(axis=1, keepdim=True)
            anchor_dists = pc_anchor - pc_anchor.mean(axis=1, keepdim=True)
            
            action_scale = torch.linalg.norm(action_dists, dim=1, keepdim=True).max(dim=2, keepdim=True).values
            anchor_scale = torch.linalg.norm(anchor_dists, dim=1, keepdim=True).max(dim=2, keepdim=True).values

            point_scale = action_scale if self.model_cfg.object_scale is not None else anchor_scale

            # Updating point clouds.
            # TODO: notation here is a bit awkward, can clean this up.
            pc_action = pc_action / point_scale * scale
            pc_anchor = pc_anchor / point_scale * scale
            pc_scale = point_scale / scale

        else:
            anchor_scale = None
            pc_scale = None

        # Updating batch frames.
        action_frame = pc_action.mean(dim=-1, keepdim=True)
        anchor_frame = pc_anchor.mean(dim=-1, keepdim=True)
        pc_action = pc_action - action_frame
        pc_anchor = pc_anchor - anchor_frame
        # rel_pos = (action_frame - anchor_frame).permute(0, 2, 1)
        rel_pos = action_frame - anchor_frame

        # Getting model kwargs.
        bs = pc_anchor.shape[0]
        model_kwargs = {
            "x": pc_anchor,
            "y": pc_action,
            "x0": pc_anchor,
            "t": torch.zeros(bs).to(pc_anchor.device),
           "rel_pos": rel_pos.permute(0, 2, 1),
        }
        output = self.network(**model_kwargs).permute(0, 2, 1)

        logits = output[..., [0]]
        residuals = output[..., 1:4]
        # vars = output[..., 4:5] # ignore last output

        # # run vars through softplus to ensure positive values
        # vars = torch.nn.functional.softplus(vars)

        # add mean residuals to points to get mean predictions
        means = residuals + pc_anchor.permute(0, 2, 1)

        # converting logits to probabilities
        probs = torch.softmax(logits, dim=1)

        # If necessary, unscale the predictions.
        if pc_scale is not None:
            pc_scale = pc_scale.to(pc_anchor.device)
            means = means * pc_scale.permute(0, 2, 1)
            residuals = residuals * pc_scale.permute(0, 2, 1)
            action_frame = action_frame * pc_scale
            anchor_frame = anchor_frame * pc_scale

        return {
            "probs": probs,
            "means": means,
            "residuals": residuals,
            "action_frame": action_frame.permute(0, 2, 1),
            "anchor_frame": anchor_frame.permute(0, 2, 1),
            "anchor_scale": anchor_scale,
            "pc_scale": pc_scale,
        }

#################################################################################
#                               GMM Likelihood Loss                             #
#################################################################################

class GMMLoss(torch.nn.Module):
    def __init__(self, cfg, eps=1e-6):
        """
        eps: value used to clamp var, for stability.
        """
        super(GMMLoss, self).__init__()
        self.pcd_scale = cfg.model.pcd_scale
        self.eps = eps
    
    def forward(self, batch, pred, var=0.00001, uniform_loss=0.0, regularize_residual=0.0):
        # Computing ground truth action mean in anchor frame.
        anchor_frame = pred["anchor_frame"]
        targets = batch["pc"].to(anchor_frame.device) - anchor_frame
        targets = targets.mean(dim=1, keepdim=True)

        # Extracting prediction weights and means.
        probs = pred["probs"]
        means = pred["means"]

        # Computing GMM likelihood loss.
        diff = targets - means

        if pred["pc_scale"] is not None:
            # var *= pred["pc_scale"].permute(0, 2, 1)
            var *= pred["anchor_scale"].permute(0, 2, 1)
        var *= self.pcd_scale
        
        point_likelihood_exps = -0.5 * torch.sum((diff ** 2) / var, dim=-1, keepdim=True)
        maxlog = point_likelihood_exps.max(dim=-2, keepdim=True).values
        point_likelihoods = torch.exp(point_likelihood_exps - maxlog)
        likelihoods = torch.sum(probs * point_likelihoods, dim=-2, keepdim=True)
        log_likelihoods = torch.log(likelihoods) + maxlog

        loss = -torch.sum(log_likelihoods)

        # Uniform loss term.
        if uniform_loss > 0.0:
            uniform_nll = -torch.sum(
                torch.log(torch.sum(point_likelihoods, dim=-2, keepdim=True)) + maxlog
            )
            loss += uniform_loss * uniform_nll
        
        # Regularization term.
        if regularize_residual > 0.0:
            residuals = pred["residuals"]
            residual_loss = torch.norm(residuals, dim=-1).mean()
            loss += regularize_residual * residual_loss
        return loss

#################################################################################
#                                 Helper Functions                              #
#################################################################################

def viz_gmm(model, dataset):
    model.eval()
    # model.to(device)
    fig = make_subplots(rows=2, cols=4,
                        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}, {"type": "scatter3d"}, {"type": "scatter3d"}], 
                            [{"type": "scatter3d"}, {"type": "scatter3d"}, {"type": "scatter3d"}, {"type": "scatter3d"}]],
                        subplot_titles=["Plot 1", "Plot2", "Plot3", "Plot4", "Plot5", "Plot6", "Plot7", "Plot8"],
    )
    data_keys = ["pc_action", "pc_anchor", "pc"]
    
    # visualize training dataset
    for i in range(2):
        item = dataset[i]
        batch = {key: value.unsqueeze(0) for key, value in item.items() if key in data_keys}

        # batch = tgd.Batch.from_data_list([train_dataset[i]]).to(device)
        with torch.no_grad():
            pred = model(batch)

        # Computing ground truth action mean in anchor frame.
        anchor_frame = pred["anchor_frame"].cpu()
        targets = batch["pc"] - anchor_frame
        targets = targets.mean(dim=1)[0].cpu().numpy()
        pc_anchor = (batch["pc_anchor"] - anchor_frame)[0].numpy()
        pc_goal = (batch["pc"] - anchor_frame)[0].numpy()
        # Extracting prediction weights and means.
        probs = pred["probs"][0].cpu().numpy().reshape((-1,))
        means = pred["means"][0].cpu().numpy()

        # prob statistics
        prob_min, prob_max, prob_med = probs.min(), probs.max(), np.median(probs)
        sorted_indices = np.argsort(probs)[::-1]

        # left plot
        fig.add_trace(
            go.Scatter3d(
                mode="markers",
                marker=dict(size=2, color="blue"),
                x=pc_anchor[:, 0],
                y=pc_anchor[:, 1],
                z=pc_anchor[:, 2],
            ), row=i+1, col=1
        )
        fig.add_trace(
            go.Scatter3d(
                mode="markers",
                marker=dict(size=2, color="green"),
                x=pc_goal[:, 0],
                y=pc_goal[:, 1],
                z=pc_goal[:, 2],
            ), row=i+1, col=1
        )
        fig.add_trace(
            go.Scatter3d(
                mode="markers",
                marker=dict(size=4, color=probs, colorscale="Viridis", cmin=0),
                x=means[:, 0],
                y=means[:, 1],
                z=means[:, 2],
            ), row=i+1, col=1
        )
        fig.add_trace(
            go.Scatter3d(
                mode="markers",
                marker=dict(size=8, color="red"),
                x=[targets[0]],
                y=[targets[1]],
                z=[targets[2]],
            ), row=i + 1, col=1
        )

        # middle plot (top 99)
        top_99_indices = sorted_indices[np.cumsum(probs[sorted_indices]) < 0.99]
        means99 = means[top_99_indices]
        probs99 = probs[top_99_indices]
        num_probs99 = len(probs99)
        fig.add_trace(
            go.Scatter3d(
                mode="markers",
                marker=dict(size=2, color="blue"),
                x=pc_anchor[:, 0],
                y=pc_anchor[:, 1],
                z=pc_anchor[:, 2],
            ), row=i+1, col=2
        )
        fig.add_trace(
            go.Scatter3d(
                mode="markers",
                marker=dict(size=2, color="green"),
                x=pc_goal[:, 0],
                y=pc_goal[:, 1],
                z=pc_goal[:, 2],
            ), row=i+1, col=2
        )
        fig.add_trace(
            go.Scatter3d(
                mode="markers",
                marker=dict(size=4, color=probs99, colorscale="Viridis", cmin=0),
                x=means99[:, 0],
                y=means99[:, 1],
                z=means99[:, 2],
            ), row=i+1, col=2
        )
        fig.add_trace(
            go.Scatter3d(
                mode="markers",
                marker=dict(size=8, color="red"),
                x=[targets[0]],
                y=[targets[1]],
                z=[targets[2]],
            ), row=i + 1, col=2
        )

        # right plot
        top_90_indices = sorted_indices[np.cumsum(probs[sorted_indices]) < 0.90]
        means90 = means[top_90_indices]
        probs90 = probs[top_90_indices]
        num_probs90 = len(probs90)
        fig.add_trace(
            go.Scatter3d(
                mode="markers",
                marker=dict(size=2, color="blue"),
                x=pc_anchor[:, 0],
                y=pc_anchor[:, 1],
                z=pc_anchor[:, 2],
            ), row=i+1, col=3
        )
        fig.add_trace(
            go.Scatter3d(
                mode="markers",
                marker=dict(size=2, color="green"),
                x=pc_goal[:, 0],
                y=pc_goal[:, 1],
                z=pc_goal[:, 2],
            ), row=i+1, col=3
        )
        fig.add_trace(
            go.Scatter3d(
                mode="markers",
                marker=dict(size=4, color=probs90, colorscale="Viridis", cmin=0),
                x=means90[:, 0],
                y=means90[:, 1],
                z=means90[:, 2],
            ), row=i+1, col=3
        )
        fig.add_trace(
            go.Scatter3d(
                mode="markers",
                marker=dict(size=8, color="red"),
                x=[targets[0]],
                y=[targets[1]],
                z=[targets[2]],
            ), row=i + 1, col=3
        )

        # top50 plot
        top_50_indices = sorted_indices[np.cumsum(probs[sorted_indices]) < 0.50]
        means50 = means[top_50_indices]
        probs50 = probs[top_50_indices]
        num_probs50 = len(probs50)
        fig.add_trace(
            go.Scatter3d(
                mode="markers",
                marker=dict(size=2, color="blue"),
                x=pc_anchor[:, 0],
                y=pc_anchor[:, 1],
                z=pc_anchor[:, 2],
            ), row=i+1, col=4
        )
        fig.add_trace(
            go.Scatter3d(
                mode="markers",
                marker=dict(size=2, color="green"),
                x=pc_goal[:, 0],
                y=pc_goal[:, 1],
                z=pc_goal[:, 2],
            ), row=i+1, col=4
        )
        fig.add_trace(
            go.Scatter3d(
                mode="markers",
                marker=dict(size=4, color=probs50, colorscale="Viridis", cmin=0),
                x=means50[:, 0],
                y=means50[:, 1],
                z=means50[:, 2],
            ), row=i+1, col=4
        )
        fig.add_trace(
            go.Scatter3d(
                mode="markers",
                marker=dict(size=8, color="red"),
                x=[targets[0]],
                y=[targets[1]],
                z=[targets[2]],
            ), row=i + 1, col=4
        )

        fig.layout.annotations[4 * i].update(text=f"Median: {prob_med:.6f} Min: {prob_min:.6f}, Max: {prob_max:.6f}")
        fig.layout.annotations[4 * i + 1].update(text=f"Top-0.99 Num Points: {num_probs99}")
        fig.layout.annotations[4 * i + 2].update(text=f"Top-0.90 Num Points: {num_probs90}")
        fig.layout.annotations[4 * i + 3].update(text=f"Top-0.50 Num Points: {num_probs50}")
    
    return fig, num_probs99, num_probs90, num_probs50
