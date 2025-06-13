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

from non_rigid.utils.hand_model import create_hand_model

import rpad.visualize_3d.plots as vpl
import plotly.graph_objects as go
import json
import trimesh
import random


class DexDataset(data.Dataset):
    def __init__(self, root, dataset_cfg, split):
        super().__init__()
        self.root = root
        self.split = split
        self.dataset_dir = self.root / self.split
        self.dataset_cfg = dataset_cfg

        # Task specific config
        self.robot_names = dataset_cfg.robot_names if dataset_cfg.robot_names is not None \
            else ['barrett', 'allegro', 'shadowhand']
        self.num_points = dataset_cfg.sample_size_action
        self.object_pc_type = dataset_cfg.object_pc_type
        self.is_train = True if "train" in self.split else False

        # Load data from file
        self.hands = {}
        for robot_name in self.robot_names:
            self.hands[robot_name] = create_hand_model(dataset_cfg.data_dir, robot_name, torch.device('cpu'))

        split_json_path = os.path.join(dataset_cfg.data_dir, f'data/CMapDataset_filtered/split_train_validate_objects.json')
        dataset_split = json.load(open(split_json_path))
        self.object_names = dataset_split['train'] if 'train' in self.split else dataset_split['validate']
        
        dataset_path = os.path.join(dataset_cfg.data_dir, f'data/CMapDataset_filtered/cmap_dataset.pt')
        metadata = torch.load(dataset_path)['metadata']
        self.metadata = [m for m in metadata if m[1] in self.object_names and m[2] in self.robot_names]
        if not self.is_train:
            self.combination = []
            for robot_name in self.robot_names:
                for object_name in self.object_names:
                    self.combination.append((robot_name, object_name))
            self.combination = sorted(self.combination)

        self.object_pcs = {}
        if self.object_pc_type != 'fixed':
            for object_name in self.object_names:
                name = object_name.split('+')
                mesh_path = os.path.join(dataset_cfg.data_dir, f'data/data_urdf/object/{name[0]}/{name[1]}/{name[1]}.stl')
                mesh = trimesh.load_mesh(mesh_path)
                object_pc, _ = mesh.sample(65536, return_index=True)
                self.object_pcs[object_name] = torch.tensor(object_pc, dtype=torch.float32)
        else:
            print("!!! Using fixed object pcs !!!")

        # getting object pc & normals for penetration loss
        self.pene_object_pc = {}
        self.pene_normals = {}
        
        if self.dataset_cfg.use_pene_loss:
            for object_name in self.object_names:
                name = object_name.split('+')
                object_path = '/home/yingyuan/DRO-Grasp/data/PointCloud/object/{name[0]}/{name[1]}.pt'.format(name=name)
                object_pc_normals = torch.load(object_path)
                self.pene_object_pc[object_name] = object_pc_normals[:, :3]
                self.pene_normals[object_name] = object_pc_normals[:, 3:]

        # Determining dataset size - if not specified, use all demos in directory once.
        if self.is_train:
            self.num_demos = len(self.metadata)
        else:
            self.num_demos = len(self.combination)
        size = self.dataset_cfg.train_size if "train" in self.split else self.dataset_cfg.val_size
        if size is not None:
            self.size = size
        else:
            self.size = self.num_demos
        print(f"Dataset size: {self.num_demos} ({self.size})")

        # Setting sample sizes.
        self.sample_size_action = self.dataset_cfg.sample_size_action
        self.sample_size_anchor = self.dataset_cfg.sample_size_anchor
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, index, return_indices=False, use_indices=None):
        """
        Args:
            return_indices: if True, return the indices used to downsample the point clouds.
            use_indices: if not None, use these indices to downsample the point clouds. If indices are provided,
                sample_size_action and sample_size_anchor are ignored.
        """
        if self.is_train:
            robot_name = random.choice(self.robot_names)
            hand = self.hands[robot_name]
            metadata_robot = [(m[0], m[1]) for m in self.metadata if m[2] == robot_name]
            target_q, object_name = random.choice(metadata_robot)

            if self.object_pc_type == 'random':
                indices = torch.randperm(65536)[:self.num_points]
                object_pc = self.object_pcs[object_name][indices]
                object_pc += torch.randn(object_pc.shape) * 0.002
            else:
                raise NotImplementedError
            
            robot_pc_target = hand.get_transformed_links_pc(target_q)[:, :3]
            initial_q = hand.get_initial_q(target_q)
            robot_pc_initial = hand.get_transformed_links_pc(initial_q)[:, :3]

            if self.dataset_cfg.use_pene_loss:
                pene_object_pc = self.pene_object_pc[object_name]
                pene_normals = self.pene_normals[object_name]
        else:
            robot_name, object_name = self.combination[index]
            hand = self.hands[robot_name]
            initial_q = hand.get_initial_q()
            robot_pc_initial = hand.get_transformed_links_pc(initial_q)[:, :3]
            robot_pc_target = torch.zeros_like(robot_pc_initial)  # same as initial, just a placeholder

            name = object_name.split('+')
            object_path = os.path.join(self.dataset_cfg.data_dir, f'data/PointCloud/object/{name[0]}/{name[1]}.pt')
            object_pc = torch.load(object_path)[:, :3]

            if self.dataset_cfg.use_pene_loss:
                pene_object_pc = self.pene_object_pc[object_name]
                pene_normals = self.pene_normals[object_name]
        
        # zero-mean the initial robot point cloud
        robot_pc_initial = robot_pc_initial - robot_pc_initial.mean(axis=0, keepdim=True)

        action_pc = robot_pc_initial.float()
        anchor_pc = object_pc.float()
        flow = robot_pc_target.float() - robot_pc_initial.float()

        # Loading segmentation masks, if available. TODO: eventually, assume these are available
        action_seg = torch.zeros_like(action_pc[:, 0]).int()
        anchor_seg = torch.ones_like(anchor_pc[:, 0]).int()

        # Initializing item dict.
        item = {
            "deform_data": {
                "deform_params": {}},
            "rigid_data": {},
        }

        # Downsample action. 
        action_pc_indices = torch.arange(action_pc.shape[0])

        # Downsample anchor.
        anchor_pc_indices = torch.arange(anchor_pc.shape[0])

        # Return indices, if specified.
        if return_indices:
            item["action_pc_indices"] = action_pc_indices
            item["anchor_pc_indices"] = anchor_pc_indices

        # Compute goal action point cloud.
        goal_action_pc = action_pc + flow

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

        item["pc_action"] = action_pc # Action points in the action frame
        item["pc_anchor"] = anchor_pc # Anchor points in the scene frame
        item["seg"] = action_seg
        item["seg_anchor"] = anchor_seg
        item["T_goal2world"] = T_goal2world.get_matrix().squeeze(0) # Transform from goal action frame to world frame
        item["T_action2world"] = T_action2world.get_matrix().squeeze(0) # Transform from action frame to world frame

        if self.dataset_cfg.use_pene_loss:
            item["pene_object_pc"] = pene_object_pc
            item["pene_normals"] = pene_normals
        else:
            item["pene_object_pc"] = None
            item["pene_normals"] = None

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

class DexDataModule(L.LightningDataModule):
    def __init__(self, batch_size, val_batch_size, num_workers, dataset_cfg):
        super().__init__()
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.dataset_cfg = dataset_cfg
        self.stage = None

        # setting root directory based on dataset type
        data_dir = os.path.expanduser(self.dataset_cfg.data_dir)
        exp_dir = (
            f"cloth={self.dataset_cfg.cloth_geometry}-{self.dataset_cfg.cloth_pose} " + \
            f"anchor={self.dataset_cfg.anchor_geometry}-{self.dataset_cfg.anchor_pose} " + \
            f"hole={self.dataset_cfg.hole} " + \
            f"robot={self.dataset_cfg.robot} " + \
            f"num_anchors={self.dataset_cfg.num_anchors}"
        )
        self.root = Path(data_dir) / self.dataset_cfg.task / exp_dir
    
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
        self.train_dataset = DexDataset(self.root, self.dataset_cfg, "train_tax3d")
        self.val_dataset = DexDataset(self.root, self.dataset_cfg, "val_tax3d")
        # self.val_ood_dataset = DexDataset(self.root, self.dataset_cfg, "val_ood_tax3d")

    def train_dataloader(self):
        return data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False if self.stage == "train" else False,
            num_workers=self.num_workers,
            collate_fn=cloth_collate_fn,
        )
    
    def val_dataloader(self):
        val_dataloader = data.DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=cloth_collate_fn,
        )
        # val_ood_dataloader = data.DataLoader(
        #     self.val_ood_dataset,
        #     batch_size=self.val_batch_size,
        #     shuffle=False,
        #     num_workers=self.num_workers,
        #     collate_fn=cloth_collate_fn,
        # )
        # return val_dataloader, val_ood_dataloader
        return (val_dataloader)
    

# custom collate function to handle deform params
def cloth_collate_fn(batch):
    # batch can contain a list of dictionaries
    # we need to convert those to a dictionary of lists
    dict_keys = ["deform_data", "rigid_data", "object_names"]
    keys = batch[0].keys()
    if batch[0]['pene_object_pc'] is None:
        dict_keys.append("pene_object_pc")
        dict_keys.append("pene_normals")
    out = {k: None for k in keys}
    for k in keys:
        if k in dict_keys:
        #if k == "deform_params":
            out[k] = [item[k] for item in batch]
        else:
            out[k] = torch.stack([item[k] for item in batch])
    return out