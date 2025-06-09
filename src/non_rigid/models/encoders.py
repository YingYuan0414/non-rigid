import torch
import torch.nn as nn
import torch_geometric.data as tgd
import numpy as np

from non_rigid.nets.pn2 import PN2Dense, PN2DenseParams
from non_rigid.nets.encoder import Encoder

from functools import partial

#################################################################################
#                               Point Cloud Encoders                            #
#################################################################################

def dgcnn_encoder(emb_dim, pretrain=None, device=torch.device('cpu')) -> nn.Module:
    encoder = Encoder(emb_dim=emb_dim)
    if pretrain is not None:
        print(f"******** Load embedding network pretrain from <{pretrain}> ********")
        encoder.load_state_dict(
            torch.load(
                f"/home/yingyuan/non-rigid/ckpt/pretrain/{pretrain}",
                map_location=device
            )
        )
    encoder.to(device)
    return encoder

def mlp_encoder(in_channels, out_channels):
    """
    MLP encoder for point clouds.
    """
    return nn.Conv1d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=True,
    )

def pn2_encoder(in_channels, out_channels, model_cfg):
    """
    PointNet++ encoder for point clouds.
    """
    pn2_params = PN2DenseParams()
    pn2_params.sa1.r = 0.2 * model_cfg.pcd_scale
    pn2_params.sa2.r = 0.4 * model_cfg.pcd_scale

    class PN2DenseWrapper(nn.Module):
        def __init__(self, in_channels, out_channels, p):
            super().__init__()
            self.pn2dense = PN2Dense(
                in_channels=in_channels - 3,
                out_channels=out_channels,
                p=p,
            )

        def forward(self, x):
            batch_size, num_channels = x.shape[0], x.shape[1]
            batch_indices = torch.arange(
                batch_size, device=x.device
            ).repeat_interleave(x.shape[2])

            if num_channels == 3:
                input_batch = tgd.Batch(
                    pos=x.permute(0, 2, 1).reshape(-1, 3), batch=batch_indices
                )
            elif num_channels > 3:
                input_batch = tgd.Batch(
                    pos=x[:, :3, :].permute(0, 2, 1).reshape(-1, 3),
                    x=x[:, 3:, :].permute(0, 2, 1).reshape(-1, num_channels - 3),
                    batch=batch_indices,
                )
            else:
                raise ValueError(f"Invalid number of input channels: {num_channels}")
            
            output = self.pn2dense(input_batch)
            output = output.reshape(batch_size, -1, output.shape[-1]).permute(0, 2, 1)
            return output
    
    return PN2DenseWrapper(in_channels=in_channels, out_channels=out_channels, p=pn2_params)

#################################################################################
#                                 Feature Encoders                              #
#################################################################################

class DisjointFeatureEncoder(nn.Module):
    """
    TODO: fill this out
    """
    def __init__(self, in_channels, hidden_size, model_cfg):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.model_cfg = model_cfg

        # Initializing point cloud encoder wrapper.
        if self.model_cfg.point_encoder == "mlp":
            encoder_fn = partial(mlp_encoder, in_channels=self.in_channels)
        elif self.model_cfg.point_encoder == "pn2":
            encoder_fn = partial(pn2_encoder, in_channels=self.in_channels, model_cfg=self.model_cfg)
        else:
            raise ValueError(f"Invalid point_encoder: {self.model_cfg.point_encoder}")

        # Creating base encoders - action (x0), anchor (y), and noised prediction (x).
        self.x_encoder = encoder_fn(out_channels=hidden_size)
        self.x0_encoder = encoder_fn(out_channels=hidden_size)
        self.y_encoder = encoder_fn(out_channels=hidden_size)

        # Creating extra feature encoders, if necessary.
        if self.model_cfg.feature:
            self.shape_encoder = encoder_fn(out_channels=hidden_size)
            self.flow_zeromean_encoder = encoder_fn(out_channels=hidden_size)
            self.x_corr_encoder = encoder_fn(out_channels=hidden_size)
            self.action_mixer = mlp_encoder(5 * hidden_size, hidden_size)
        else:
            self.action_mixer = mlp_encoder(2 * hidden_size, hidden_size)
    
    def forward(self, x, y, x0):
        """
        TODO: fill this out
        """
        if self.model_cfg.type == "flow":
            x_flow = x
            x_recon = x + x0
        else:
            x_flow = x - x0
            x_recon = x
        
        # Encode base features - action (x0), anchor (y), and noised prediction (x).
        x_enc = self.x_encoder(x)
        x0_enc = self.x0_encoder(x0)
        y_enc = self.y_encoder(y).permute(0, 2, 1)

        # Encode extra features, if necessary.
        if self.model_cfg.feature:
            shape_enc = self.shape_encoder(
                x_recon - torch.mean(x_recon, dim=2, keepdim=True)
            )
            flow_zeromean_enc = self.flow_zeromean_encoder(
                x_flow - torch.mean(x_flow, dim=2, keepdim=True)
            )
            x_corr_enc = self.x_corr_encoder(
                x_recon if self.model_cfg.type == "flow" else x_flow
            )
            action_features = [x_enc, x0_enc, shape_enc, flow_zeromean_enc, x_corr_enc]
        else:
            action_features = [x_enc, x0_enc]

        # Compress action features to hidden size through action mixer.
        x_enc = torch.cat(action_features, dim=1)
        x_enc = self.action_mixer(x_enc).permute(0, 2, 1)

        return x_enc, y_enc

class JointFeatureEncoder(nn.Module):
    """
    TODO: fill this out
    """
    def __init__(self, in_channels, hidden_size, model_cfg):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.model_cfg = model_cfg

        # Initializing point cloud encoder wrapper.
        if self.model_cfg.point_encoder == "mlp":
            encoder_fn = partial(mlp_encoder, in_channels=self.in_channels, out_channels=hidden_size)
        elif self.model_cfg.point_encoder == "pn2":
            encoder_fn = partial(pn2_encoder, in_channels=self.in_channels, out_channels=hidden_size, model_cfg=self.model_cfg)
        elif self.model_cfg.point_encoder == "dgcnn":  # hidden_size should be 512 to match pre-trained DRO model.
            self.action_encoder = dgcnn_encoder(emb_dim=hidden_size, pretrain="pretrain_3robots_128.pth", device='cuda')
            # self.pred_encoder = dgcnn_encoder(emb_dim=hidden_size, device='cuda')
            self.pred_encoder = pn2_encoder(in_channels=self.in_channels, out_channels=hidden_size, model_cfg=self.model_cfg)
        else:
            raise ValueError(f"Invalid point_encoder: {self.model_cfg.point_encoder}")

        # Creating base encoders - action-frame, and prediction-frame.
        if self.model_cfg.point_encoder != "dgcnn":
            self.action_encoder = encoder_fn()
            self.pred_encoder = encoder_fn()

        # Creating extra feature encoders, if necessary.
        if self.model_cfg.feature:
            self.feature_encoder = pn2_encoder(in_channels=9, out_channels=hidden_size, model_cfg=self.model_cfg)
            self.action_mixer = mlp_encoder(3 * hidden_size, hidden_size)
        else:
            self.action_mixer = mlp_encoder(2 * hidden_size, hidden_size)
    
    def forward(self, x, y, x0):
        """
        TODO: fill this out
        """
        if self.model_cfg.type == "flow":
            x_flow = x
            x_recon = x + x0
        else:
            x_flow = x - x0
            x_recon = x
        
        # Encode base features - action-frame, and prediction frame.
        action_size = x0.shape[-1]

        if self.model_cfg.point_encoder != "dgcnn":
            x0_onehot = torch.zeros((x0.shape[0], 3, action_size), device=x0.device)
            x0_onehot[:, 2, :] = 1
            x0_wh = torch.cat([x0, x0_onehot], dim=1)  
        else:
            x0_wh = x0

        action_enc = self.action_encoder(x0_wh[:, :self.in_channels, :])  # paper: object embedding o_j
        if self.model_cfg.point_encoder == "dgcnn":
            action_enc = action_enc.detach()
        
        if self.model_cfg.point_encoder != "dgcnn":
            x_recon_onehot = torch.zeros((x_recon.shape[0], 3, action_size), device=x_recon.device)
            x_recon_onehot[:, 0, :] = 1
            y_onehot = torch.zeros((y.shape[0], 3, y.shape[2]), device=y.device)
            y_onehot[:, 1, :] = 1
            x_recon_wh = torch.cat([x_recon, x_recon_onehot], dim=1)
            y_wh = torch.cat([y, y_onehot], dim=1) 
        else:
            x_recon_wh = x_recon
            y_wh = y

        pred_enc = self.pred_encoder(torch.cat([x_recon_wh[:, :self.in_channels, :], y_wh[:, :self.in_channels, :]], dim=-1))  # paper: reconstructed placement -> reconstruction embedding f_i
        
        action_pred_enc, anchor_pred_enc = pred_enc[:, :, :action_size], pred_enc[:, :, action_size:]
        anchor_pred_enc = anchor_pred_enc.permute(0, 2, 1)

        # Encode extra features, if necessary.
        if self.model_cfg.feature:
            shape = x_recon - torch.mean(x_recon, dim=2, keepdim=True)
            flow_zeromean = x_flow - torch.mean(x_flow, dim=2, keepdim=True)
            feature_enc = self.feature_encoder(
                torch.cat([shape, x_flow, flow_zeromean], dim=1)  # paper: displacements -> deformation embedding d_k
            )
            action_features = [action_enc, action_pred_enc, feature_enc]
        else:
            action_features = [action_enc, action_pred_enc]
        
        # Compress action features to hidden size through action mixer.
        x_enc = torch.cat(action_features, dim=1)
        x_enc = self.action_mixer(x_enc).permute(0, 2, 1)

        return x_enc, anchor_pred_enc