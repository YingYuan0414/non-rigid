# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from timm.layers import use_fused_attn
from typing import Optional
from functools import partial

from non_rigid.models.encoders import DisjointFeatureEncoder, JointFeatureEncoder, mlp_encoder, pn2_encoder

torch.set_printoptions(precision=8, sci_mode=True)

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings

#################################################################################
#                             Custom Attention Layers                           #
#################################################################################

class CrossAttention(nn.Module):
    """
    Cross attention layer adapted from
    https://github.com/pprp/timm/blob/e9aac412de82310e6905992e802b1ee4dc52b5d1/timm/models/crossvit.py#L132
    """

    def __init__(
        self,
        dim_x: int,
        dim_y: int,
        num_heads: int = 4,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        assert dim_x % num_heads == 0, "dim x must be divisible by num_heads"
        head_dim = dim_x // num_heads
        self.scale = head_dim**-0.5
        self.fused_attn = use_fused_attn()

        self.wq = nn.Linear(dim_x, dim_x, bias=qkv_bias)
        self.wk = nn.Linear(dim_y, dim_x, bias=qkv_bias)
        self.wv = nn.Linear(dim_y, dim_x, bias=qkv_bias)
        self.proj = nn.Linear(dim_x, dim_x)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        B, N, Cx = x.shape
        # _, _, Cy = y.shape
        _, Ny, Cy = y.shape
        q = (
            self.wq(x)
            .reshape(B, N, self.num_heads, Cx // self.num_heads)
            .transpose(1, 2)
        )
        k = (
            self.wk(y)
            .reshape(B, Ny, self.num_heads, Cx // self.num_heads)
            .transpose(1, 2)
        )
        v = (
            self.wv(y)
            .reshape(B, Ny, self.num_heads, Cx // self.num_heads)
            .transpose(1, 2)
        )

        if self.fused_attn:
            x = (
                F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    dropout_p=self.attn_drop.p if self.training else 0.0,
                )
                .transpose(1, 2)
                .reshape(B, N, Cx)
            )
        else:
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, Cx)

        # attn = (q @ k.transpose(-2, -1)) * self.scale
        # attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)
        # x = (attn @ v).transpose(1, 2).reshape(B, N, Cx)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

#################################################################################
#                                 Core DiT Layers                               #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class DiTCrossBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning and scene cross attention.
    """

    def __init__(
        self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, **block_kwargs
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.self_attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = CrossAttention(
            dim_x=hidden_size,
            dim_y=hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            **block_kwargs,
        )
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )

    def forward(self, x, y, c):
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mca,
            scale_mca,
            gate_mca,
            shift_x,
            scale_x,
            gate_x,
        ) = self.adaLN_modulation(c).chunk(9, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.self_attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mca.unsqueeze(1) * self.cross_attn(
            modulate(self.norm2(x), shift_mca, scale_mca), y
        )
        x = x + gate_x.unsqueeze(1) * self.mlp(
            modulate(self.norm3(x), shift_x, scale_x)
        )
        return x

class DiTRefBlock(nn.Module):
    """
    A simplified DiT block for the reference branch.
    This block processes a single reference token by attending to scene tokens,
    using only cross attention. Self-attention is omitted because with one token it
    doesn't provide additional interactions.
    
    The block still uses layer normalization and an MLP for refinement, and it accepts
    a conditioning vector (e.g. t_emb) to modulate its operations.
    """
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, **kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.cross_attn = CrossAttention(
            dim_x=hidden_size,
            dim_y=hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            **kwargs,
        )
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=nn.GELU,
            drop=0,
        )
        # We can include a simple modulation block if desired.
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        )

    def forward(self, ref_token: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ref_token: Tensor of shape (B, 1, hidden_size) representing the reference token.
            y: Scene (anchor) tokens of shape (B, N, hidden_size).
            c: Conditioning vector (e.g. t_emb) of shape (B, hidden_size).
        Returns:
            Updated reference token of shape (B, 1, hidden_size).
        """
        # Compute modulation parameters from the conditioning vector.
        # Here we split c into three parts: shift, scale, and gate.
        shift, scale, gate = self.adaLN_modulation(c).chunk(3, dim=1)
        
        # Normalize the reference token and modulate it.
        ref_norm = self.norm1(ref_token)
        modulated_ref = modulate(ref_norm, shift, scale)
        
        # Let the reference token attend to the scene tokens via cross attention.
        attn_out = self.cross_attn(modulated_ref, y)
        x = ref_token + gate.unsqueeze(1) * attn_out
        
        # Further refine with an MLP.
        x = x + self.mlp(self.norm2(x))
        return x

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class FinalLayer_s(nn.Module):
    """
    Final layer to predict shape (xs) component
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class FinalLayer_r(nn.Module):
    """
    Final layer to predict mean (xr) component
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        # x shape: [8, 512, 128]
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        
        x = x.mean(dim=1, keepdim=True)  # [8, 1, 128]
        
        x = self.linear(x)  # [8, 1, 6]
        return x

#################################################################################
#                                 Core DiT Models                               #
#################################################################################

class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)
        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

class LinearRegressionModel(nn.Module):
    """
    Linear regression baseline, with attention - no diffusion.
    """
    def __init__(
            self,
            in_channels=3,
            hidden_size=1152,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            model_cfg=None,
    ):
        super().__init__()
        self.model_cfg = model_cfg
        self.out_channels = 3

        # initializing embedder for action point cloud
        self.x_embedder = nn.Conv1d(
            in_channels,
            hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        # initializing embedder for anchor point cloud
        self.y_embedder = nn.Conv1d(
            in_channels,
            hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        class LinearCrossBlock(nn.Module):
            """
            Cross attention block for linear regression model.
            """
            def __init__(self, hidden_size, num_heads, mlp_ratio):
                super().__init__()
                self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
                self.cross_attn = CrossAttention(
                    dim_x=hidden_size,
                    dim_y=hidden_size,
                    num_heads=num_heads,
                    qkv_bias=True,
                )
                mlp_hidden_dim = int(hidden_size * mlp_ratio)
                approx_gelu = lambda: nn.GELU(approximate="tanh")
                self.mlp = Mlp(
                    in_features=hidden_size,
                    hidden_features=mlp_hidden_dim,
                    act_layer=approx_gelu,
                    drop=0,
                )

            def forward(self, x, y):
                x = self.cross_attn(self.norm(x), y)
                x = self.mlp(x)
                return x
            
        class LinearFinalLayer(nn.Module):
            """
            Final layer of the linear regression model.
            """
            def __init__(self, hidden_size, out_channels):
                super().__init__()
                self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
                self.linear = nn.Linear(hidden_size, out_channels, bias=True)

            def forward(self, x):
                x = self.norm_final(x)
                x = self.linear(x)
                return x


        # TODO: DUPLICATE THE CROSS ATTENTION LAYER BASED ON DEPTH
        self.blocks = nn.ModuleList(
            [
                LinearCrossBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )

        self.final_layer = LinearFinalLayer(hidden_size, self.out_channels)
        self.initialize_weights()
    
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        self.apply(_basic_init)

        # Initialize x_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of linear regression model.
        """
        # encode x and y
        x = torch.transpose(self.x_embedder(x), -1, -2)
        y = torch.transpose(self.y_embedder(y), -1, -2)
        # forward pass through cross attention blocks
        for block in self.blocks:
            x = block(x, y)
        
        # final layer
        x = self.final_layer(x)
        x = torch.transpose(x, -1, -2)
        return x
    
class DiT_PointCloud_Cross(nn.Module):
    """
    Diffusion Transformer adapted for point cloud inputs. Uses object-centric cross attention, 
    and per-feature encoding.
    """
    def __init__(
            self,
            in_channels=3,
            hidden_size=1152,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            learn_sigma=True,
            model_cfg=None,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        # self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.out_channels = 6 if learn_sigma else 3
        self.num_heads = num_heads
        self.model_cfg = model_cfg

        # Point cloud feature encoder.
        if self.model_cfg.joint_encode:
            self.feature_encoder = JointFeatureEncoder(in_channels, hidden_size, model_cfg)
        else:
            self.feature_encoder = DisjointFeatureEncoder(in_channels, hidden_size, model_cfg)
        
        # Timestamp embedding.
        self.t_embedder = TimestepEmbedder(hidden_size)

        # Relative position embedding, if enabled.
        if self.model_cfg.rel_pos:
            self.rel_pos_embedder = nn.Sequential(
                nn.Linear(in_channels, hidden_size, bias=True),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size, bias=True),
            )
        else:
            self.rel_pos_embedder = None

        # DiT blocks.
        self.blocks = nn.ModuleList(
            [
                DiTCrossBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )

        # Final layer; functionally setting patch size to 1 for a point cloud.
        self.final_layer = FinalLayer(hidden_size, 1, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Initialize relative position embedding MLP, if enabled:
        if self.rel_pos_embedder is not None:
            nn.init.normal_(self.rel_pos_embedder[0].weight, std=0.02)
            nn.init.normal_(self.rel_pos_embedder[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            y: torch.Tensor,
            x0: torch.Tensor,
            rel_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of DiT with scene cross attention.

        Args:
            x (torch.Tensor): (B, D, N) tensor of batched current timestep x (e.g. noised action) features
            t (torch.Tensor): (B,) tensor of diffusion timesteps
            y (torch.Tensor): (B, D, N) tensor of un-noised scene (e.g. anchor) features
            x0 (torch.Tensor): (B, D, N) tensor of un-noised x (e.g. action) features
        """
        # Encode action and anchor features.
        x_enc, y_enc = self.feature_encoder(x=x, y=y, x0=x0)

        # Timestep embedding.
        t_emb = self.t_embedder(t)

        # Relative position embedding, if enabled.
        if self.model_cfg.rel_pos:
            assert rel_pos is not None, "Relative position embedding requires rel_pos tensor."
            rel_pos_emb = self.rel_pos_embedder(rel_pos.squeeze(1))
            c = t_emb + rel_pos_emb
        else:
            c = t_emb

        # Forward pass through DiT blocks.
        for block in self.blocks:
            x_enc = block(x_enc, y_enc, c)

        # Final layer.
        out = self.final_layer(x_enc, c)
        out = out.permute(0, 2, 1)
        return out

#################################################################################
#                                TAX3Dv2 Models                                 #
#################################################################################

class TAX3Dv2_MuFrame_DiT(nn.Module):
    """
    Diffusion Transformer adapted for point cloud inputs. Uses object-centric cross attention, 
    and joint-feature encoding.
    """
    def __init__(
            self,
            in_channels=3,
            hidden_size=1152,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            learn_sigma=True,
            model_cfg=None,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        # self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.out_channels = 6 if learn_sigma else 3
        self.num_heads = num_heads
        self.model_cfg = model_cfg

        # Point cloud feature encoder.
        if self.model_cfg.joint_encode:
            self.feature_encoder = JointFeatureEncoder(in_channels, hidden_size, model_cfg)
        else:
            self.feature_encoder = DisjointFeatureEncoder(in_channels, hidden_size, model_cfg)

        # Learnable frame embedding.
        self.ref_frame_token = nn.Parameter(torch.randn(1, 1, hidden_size))

        # Timestamp embedding.
        self.t_embedder = TimestepEmbedder(hidden_size)

        # Relative position embedding, if enabled.
        if self.model_cfg.rel_pos:
            self.rel_pos_embedder = nn.Sequential(
                nn.Linear(in_channels, hidden_size, bias=True),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size, bias=True),
            )
        else:
            self.rel_pos_embedder = None

        # DiT blocks.
        self.blocks = nn.ModuleList(
            [
                DiTCrossBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )

        # Final layers; functionally setting patch size to 1 for a point cloud.
        self.final_layer_r = FinalLayer(hidden_size, 1, self.out_channels)
        self.final_layer_s = FinalLayer(hidden_size, 1, self.out_channels)    
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Initialize relative position embedding MLP, if enabled:
        if self.rel_pos_embedder is not None:
            nn.init.normal_(self.rel_pos_embedder[0].weight, std=0.02)
            nn.init.normal_(self.rel_pos_embedder[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer_r.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer_r.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer_r.linear.weight, 0)
        nn.init.constant_(self.final_layer_r.linear.bias, 0)

        nn.init.constant_(self.final_layer_s.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer_s.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer_s.linear.weight, 0)
        nn.init.constant_(self.final_layer_s.linear.bias, 0)

    def forward(
            self,
            xr_t: torch.Tensor,
            xs_t: torch.Tensor,
            t: torch.Tensor,
            y: torch.Tensor,
            x0: torch.Tensor,
            rel_pos: Optional[torch.Tensor] = None,
            finetune_frame: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of DiT with scene cross attention.

        Args:
            x (torch.Tensor): (B, D, N) tensor of batched current timestep x (e.g. noised action) features
            t (torch.Tensor): (B,) tensor of diffusion timesteps
            y (torch.Tensor): (B, D, N) tensor of un-noised scene (e.g. anchor) features
            x0 (torch.Tensor): (B, D, N) tensor of un-noised x (e.g. action) features
        """        
        # Dynamically center anchor in mu-frame.
        if finetune_frame is None:
            y = y - xr_t
            # Encode action and anchor features.
            x_enc, y_enc = self.feature_encoder(x=xs_t, y=y, x0=x0)
        else:
            y = y - finetune_frame
            x = xr_t + xs_t
            # Encode action and anchor features.
            x_enc, y_enc = self.feature_encoder(x=x, y=y, x0=x0)

        # Concatenating reference frame token to the action features.
        x_enc = torch.cat(
            [x_enc, self.ref_frame_token.expand(x_enc.shape[0], -1, -1)], dim=1
        )

        # Timestep embedding.
        t_emb = self.t_embedder(t)

        # Relative position embedding, if enabled.
        if self.model_cfg.rel_pos:
            assert rel_pos is not None, "Relative position embedding requires rel_pos tensor."
            rel_pos_emb = self.rel_pos_embedder(rel_pos.squeeze(1))
            c = t_emb + rel_pos_emb
        else:
            c = t_emb

        # Forward pass through DiT blocks.
        for block in self.blocks:
            x_enc = block(x_enc, y_enc, c)

        # Final layers.
        xr_out = x_enc[:, -1:, :]
        xs_out = x_enc[:, :-1, :]
        xr_out = self.final_layer_r(xr_out, c).permute(0, 2, 1)
        xs_out = self.final_layer_s(xs_out, c).permute(0, 2, 1)

        return xr_out, xs_out

class TAX3Dv2_FixedFrame_Token_DiT(nn.Module):
    """
    Diffusion Transformer adapted for point cloud inputs. Uses object-centric cross attention, 
    and joint-feature encoding.
    """
    def __init__(
            self,
            in_channels=3,
            hidden_size=1152,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            learn_sigma=True,
            model_cfg=None,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        # self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.out_channels = 6 if learn_sigma else 3
        self.num_heads = num_heads
        self.model_cfg = model_cfg

        # Point cloud feature encoder.
        if self.model_cfg.joint_encode:
            self.feature_encoder = JointFeatureEncoder(in_channels, hidden_size, model_cfg)
        else:
            self.feature_encoder = DisjointFeatureEncoder(in_channels, hidden_size, model_cfg)

        # Learnable frame embedding.
        self.ref_frame_token = nn.Parameter(torch.randn(1, 1, hidden_size))

        # Timestamp embedding.
        self.t_embedder = TimestepEmbedder(hidden_size)

        # Relative position embedding, if enabled.
        if self.model_cfg.rel_pos:
            self.rel_pos_embedder = nn.Sequential(
                nn.Linear(in_channels, hidden_size, bias=True),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size, bias=True),
            )
        else:
            self.rel_pos_embedder = None

        # DiT blocks.
        self.blocks = nn.ModuleList(
            [
                DiTCrossBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )

        # Final layer; functionally setting patch size to 1 for a point cloud.
        self.final_layer_r = FinalLayer(hidden_size, 1, self.out_channels)
        self.final_layer_s = FinalLayer(hidden_size, 1, self.out_channels)
        self.initialize_weights()
    
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Initialize relative position embedding MLP, if enabled:
        if self.rel_pos_embedder is not None:
            nn.init.normal_(self.rel_pos_embedder[0].weight, std=0.02)
            nn.init.normal_(self.rel_pos_embedder[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer_r.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer_r.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer_r.linear.weight, 0)
        nn.init.constant_(self.final_layer_r.linear.bias, 0)

        nn.init.constant_(self.final_layer_s.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer_s.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer_s.linear.weight, 0)
        nn.init.constant_(self.final_layer_s.linear.bias, 0)

    def forward(
            self,
            xr_t: torch.Tensor,
            xs_t: torch.Tensor,
            t: torch.Tensor,
            y: torch.Tensor,
            x0: torch.Tensor,
            rel_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of DiT with scene cross attention.

        Args:
            x (torch.Tensor): (B, D, N) tensor of batched current timestep x (e.g. noised action) features
            t (torch.Tensor): (B,) tensor of diffusion timesteps
            y (torch.Tensor): (B, D, N) tensor of un-noised scene (e.g. anchor) features
            x0 (torch.Tensor): (B, D, N) tensor of un-noised x (e.g. action) features
        """
        x = xr_t + xs_t
        # Encode action and anchor features.
        x_enc, y_enc = self.feature_encoder(x=x, y=y, x0=x0)

        # Concatenating reference frame token to the action features.
        x_enc = torch.cat(
            [x_enc, self.ref_frame_token.expand(x_enc.shape[0], -1, -1)], dim=1
        )

        # Timestep embedding.
        t_emb = self.t_embedder(t)

        # Relative position embedding, if enabled.
        if self.model_cfg.rel_pos:
            assert rel_pos is not None, "Relative position embedding requires rel_pos tensor."
            rel_pos_emb = self.rel_pos_embedder(rel_pos.squeeze(1))
            c = t_emb + rel_pos_emb
        else:
            c = t_emb

        # Forward pass through DiT blocks.
        for block in self.blocks:
            x_enc = block(x_enc, y_enc, c)

        # Final layers.
        xr_out = x_enc[:, -1:, :]
        xs_out = x_enc[:, :-1, :]
        xr_out = self.final_layer_r(xr_out, c).permute(0, 2, 1)        
        xs_out = self.final_layer_s(xs_out, c).permute(0, 2, 1)

        return xr_out, xs_out

class TAX3Dv2_FixedFrame_Dual_DiT(nn.Module):
    """
    Diffusion Transformer adapted for point cloud inputs. Uses object-centric cross attention, 
    and joint-feature encoding.
    """
    def __init__(
            self,
            in_channels=3,
            hidden_size=1152,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            learn_sigma=True,
            model_cfg=None,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        # self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.out_channels = 6 if learn_sigma else 3
        self.num_heads = num_heads
        self.model_cfg = model_cfg



        x_encoder_hidden_dims = hidden_size // 2
        
        self.x_embedder = nn.Conv1d(
            in_channels,
            x_encoder_hidden_dims//2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        
        self.xr_embedder = nn.Conv1d(
                in_channels,
                x_encoder_hidden_dims//2,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
        )

        self.xs_embedder = nn.Conv1d(
                in_channels,
                x_encoder_hidden_dims//2,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
        )

        # Encoder for y features
        self.y_embedder = nn.Conv1d(
            in_channels,
            hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        

        # Encoder for x0 features
        self.x0_embedder = nn.Conv1d(
            in_channels,
            x_encoder_hidden_dims,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        
        # Timestamp embedding
        self.t_embedder = TimestepEmbedder(hidden_size)

        # DiT blocks
        block_fn = DiTCrossBlock
        self.blocks_r = nn.ModuleList(
            [
                block_fn(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )
        self.blocks_s = nn.ModuleList(
            [
                block_fn(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )
        # functionally setting patch size to 1 for a point cloud
        self.final_layer_r = FinalLayer_r(hidden_size, 1, self.out_channels)
        self.final_layer_s = FinalLayer_s(hidden_size, 1, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize x_embed like nn.Linear (instead of nn.Conv2d):
        '''
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)
        '''
        wr = self.xr_embedder.weight.data
        nn.init.xavier_uniform_(wr.view([wr.shape[0], -1]))
        nn.init.constant_(self.xr_embedder.bias, 0)
        ws = self.xs_embedder.weight.data
        nn.init.xavier_uniform_(ws.view([ws.shape[0], -1]))
        nn.init.constant_(self.xs_embedder.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks_r:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        for block in self.blocks_s:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer_r.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer_r.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer_r.linear.weight, 0)
        nn.init.constant_(self.final_layer_r.linear.bias, 0)

        nn.init.constant_(self.final_layer_s.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer_s.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer_s.linear.weight, 0)
        nn.init.constant_(self.final_layer_s.linear.bias, 0)
 
    def forward(
            self,
            xr_t: torch.Tensor,
            xs_t: torch.Tensor,
            t: torch.Tensor,
            y: torch.Tensor,
            x0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of DiT with scene cross attention.

        Args:
            x (torch.Tensor): (B, D, N) tensor of batched current timestep x (e.g. noised action) features
            t (torch.Tensor): (B,) tensor of diffusion timesteps
            y (torch.Tensor): (B, D, N) tensor of un-noised scene (e.g. anchor) features
            x0 (Optional[torch.Tensor]): (B, D, N) tensor of un-noised x (e.g. action) features
        """
        # noise-centering, if enabled
        '''
        if self.model_cfg.type == "point":
            delta_center = torch.mean(x, dim=2, keepdim=True)
            x = x - delta_center
            y = y - delta_center
            x0 = x0 - delta_center
        elif self.model_cfg.type == "flow":
            reconstruction = x + x0
            delta_center = torch.mean(reconstruction, dim=2, keepdim=True)
            x = x - delta_center
            y = y - delta_center
            x0 = x0 - delta_center
        '''
        
        # encode x, y, x0 features
        
        x = xr_t + xs_t
        recon_emb = self.x_embedder(x)
        
        xr_emb = self.xr_embedder(xr_t)
        xr_emb_exp = xr_emb.expand(-1, -1, recon_emb.size(-1))  # [B, hidden_half, N]
        xr_emb = torch.cat([xr_emb_exp, recon_emb], dim=1)

        xs_emb = self.xs_embedder(xs_t)
        xs_emb = torch.cat([xs_emb, recon_emb], dim=1)


        x0_emb = self.x0_embedder(x0)
        xr_emb = torch.cat([xr_emb, x0_emb], dim=1)
        xs_emb = torch.cat([xs_emb, x0_emb], dim=1)

        y_emb = self.y_embedder(y)
        y_emb = y_emb.permute(0, 2, 1)

        xr = xr_emb.permute(0, 2, 1)
        xs = xs_emb.permute(0, 2, 1)

        # timestep embedding
        t_emb = self.t_embedder(t)

        # forward pass through DiT blocks
        for block in self.blocks_r:
            xr = block(xr, y_emb, t_emb)
        for block in self.blocks_s:
            xs = block(xs, y_emb, t_emb)
        # (8,512, 128)

        # final layer
        xs = self.final_layer_s(xs, t_emb) # (8,512, 6)

        xr = self.final_layer_r(xr, t_emb) # (8,1, 6)

        xs = xs.permute(0, 2, 1)
        xr = xr.permute(0, 2, 1)

        return xr, xs

#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}