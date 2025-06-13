import os
from pathlib import Path

import numpy as np
import lightning as L
import torch
import torch.utils.data as data

from pytorch3d.transforms import Transform3d, Translate
from pytorch3d.transforms import matrix_to_quaternion, matrix_to_rotation_6d

from non_rigid.utils.transform_utils import random_se3
from non_rigid.utils.pointcloud_utils import downsample_pcd 
from non_rigid.utils.augmentation_utils import plane_occlusion

import rpad.visualize_3d.plots as vpl
import plotly.graph_objects as go

class ArticulatedDataset(data.Dataset):
    def __init__(self, root, dataset_cfg, split):
        super().__init__()
        self.root = root
        self.split = split
        self.dataset_dir = self.root / self.split
        self.num_demos = int(len(os.listdir(self.dataset_dir)))
        self.dataset_cfg = dataset_cfg

        # Determining dataset size - if not specified, use all demos in directory once.
        size = self.dataset_cfg.train_size if "train" in self.split else self.dataset_cfg.val_size
        if size is not None:
            self.size = size
        else:
            self.size = self.num_demos

        # Setting sample sizes.
        self.sample_size_action = self.dataset_cfg.sample_size_action
        self.sample_size_anchor = self.dataset_cfg.sample_size_anchor

        # Processing file list.
        self.file_list = os.listdir(self.dataset_dir)
        self.num_demos = len(self.file_list)

    def __len__(self):
        return self.size
    
    def __getitem__(self, index, return_indices=False, use_indices=None):
        """
        Args:
            return_indices: if True, return the indices used to downsample the point clouds.
            use_indices: if not None, use these indices to downsample the point clouds. If indices are provided,
                sample_size_action and sample_size_anchor are ignored.
        """
        # Loop over the dataset multiple times - allows for arbitrary dataset and batch size.
        file_index = index % self.num_demos

        # Load data.
        demo = np.load(self.dataset_dir / self.file_list[file_index], allow_pickle=True)
        action_pc = torch.as_tensor(demo["pc_init"]).float()
        anchor_pc = action_pc.clone()
        goal_action_pc = torch.as_tensor(demo["pc_goal"]).float()

        # Initializing item dict.
        item = {}

        # Downsample action. 
        if use_indices is not None and "action_pc_indices" in use_indices:
            action_pc_indices = use_indices["action_pc_indices"]
            action_pc = action_pc[action_pc_indices]
            goal_action_pc = goal_action_pc[action_pc_indices]
        elif self.sample_size_action > 0 and action_pc.shape[0] > self.sample_size_action:
            action_pc, action_pc_indices = downsample_pcd(action_pc.unsqueeze(0), self.sample_size_action, type=self.dataset_cfg.downsample_type)
            action_pc_indices = action_pc_indices.squeeze(0)
            action_pc = action_pc.squeeze(0)
            goal_action_pc = goal_action_pc[action_pc_indices]
        else:
            action_pc_indices = torch.arange(action_pc.shape[0])

        # Downsample anchor.
        if use_indices is not None and "anchor_pc_indices" in use_indices:
            anchor_pc_indices = use_indices["anchor_pc_indices"]
            anchor_pc = anchor_pc[anchor_pc_indices]
        elif self.sample_size_anchor > 0 and anchor_pc.shape[0] > self.sample_size_anchor:
            anchor_pc, anchor_pc_indices = downsample_pcd(anchor_pc.unsqueeze(0), self.sample_size_anchor, type=self.dataset_cfg.downsample_type)
            anchor_pc_indices = anchor_pc_indices.squeeze(0)
            anchor_pc = anchor_pc.squeeze(0)
        else:
            anchor_pc_indices = torch.arange(anchor_pc.shape[0])
        
        # Return indices, if specified.
        if return_indices:
            item["action_pc_indices"] = action_pc_indices
            item["anchor_pc_indices"] = anchor_pc_indices
        
        # Apply scene-level augmentation.
        T = random_se3(
            N=1,
            rot_var=self.dataset_cfg.rotation_variance,
            trans_var=self.dataset_cfg.translation_variance,
            rot_sample_method=self.dataset_cfg.scene_transform_type,
        )
        action_pc = T.transform_points(action_pc)
        anchor_pc = T.transform_points(anchor_pc)
        goal_action_pc = T.transform_points(goal_action_pc)

        # Center point clouds in scene frame.
        scene_center = torch.cat([action_pc, anchor_pc], dim=0).mean(axis=0)
        goal_action_pc = goal_action_pc - scene_center
        anchor_pc = anchor_pc - scene_center
        action_pc = action_pc - scene_center

        # Update item.
        T_goal2world = Translate(scene_center.unsqueeze(0)).compose(T.inverse())
        T_action2world = Translate(scene_center.unsqueeze(0)).compose(T.inverse())

        goal_flow = goal_action_pc - action_pc

        # Creating segmentation masks.
        action_seg = torch.zeros_like(action_pc[:, 0]).int()
        anchor_seg = torch.ones_like(anchor_pc[:, 0]).int()

        item["pc_action"] = action_pc # Action points in the action frame
        item["pc_anchor"] = anchor_pc # Anchor points in the scene frame
        item["seg"] = action_seg
        item["seg_anchor"] = anchor_seg
        item["T_goal2world"] = T_goal2world.get_matrix().squeeze(0) # Transform from goal action frame to world frame
        item["T_action2world"] = T_action2world.get_matrix().squeeze(0) # Transform from action frame to world frame

        # Training-specific labels.
        # TODO: eventually, rename this key to "point"
        item["pc"] = goal_action_pc # Ground-truth goal action points in the scene frame
        item["flow"] = goal_flow # Ground-truth flow (cross-frame) to action points
        
        if self.dataset_cfg.pred_frame == "noisy_goal":
            # "Simulate" the GMM prediction as noisy goal.
            goal_center = goal_action_pc.mean(axis=0)
            item["noisy_goal"] = goal_center + self.dataset_cfg.noisy_goal_scale * torch.normal(mean=torch.zeros(3), std=torch.ones(3))

        return item

    def set_eval_mode(self, eval_mode):
        return

class ArticulatedDataModule(L.LightningModule):
    def __init__(self, batch_size, val_batch_size, num_workers, dataset_cfg):
        super().__init__()
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.dataset_cfg = dataset_cfg
        self.stage = None

        # setting root directory
        data_dir = os.path.expanduser(self.dataset_cfg.data_dir)
        self.root = Path(data_dir)
    
    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str = "fit"):
        self.stage = stage

        # if not in train mode, don't use rotation augmentations
        if self.stage != "fit":
            print("-------Turning off rotation augmentation for validation/inference.-------")
            self.dataset_cfg.scene_transform_type = "identity"
            self.dataset_cfg.rotation_variance = 0.0
            self.dataset_cfg.translation_variance = 0.0

        # initializing datasets
        self.train_dataset = ArticulatedDataset(self.root, self.dataset_cfg, "train")
        self.val_dataset = ArticulatedDataset(self.root, self.dataset_cfg, "val")
        self.val_ood_dataset = ArticulatedDataset(self.root, self.dataset_cfg, "val_ood")
    
    def train_dataloader(self):
        return data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False if self.stage == "train" else False,
            num_workers=self.num_workers,
        )
    
    def val_dataloader(self):
        val_dataloader = data.DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        # val_ood_dataloader = data.DataLoader(
        #     self.val_ood_dataset,
        #     batch_size=self.val_batch_size,
        #     shuffle=False,
        #     num_workers=self.num_workers,
        # )
        # return val_dataloader, val_ood_dataloader
        return (val_dataloader)
    