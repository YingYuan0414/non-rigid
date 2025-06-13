import numpy as np
from pathlib import Path
import os

import torch 

from rpad.partnet_mobility_utils.data import PMObject
from rpad.partnet_mobility_utils.dataset import (
    UMPNET_TRAIN_TRAIN_OBJS as train_objs,
    UMPNET_TRAIN_TEST_OBJS as val_objs,
    UMPNET_TEST_OBJS as val_ood_objs,
)
from rpad.partnet_mobility_utils.render.pybullet import PybulletRenderer
from rpad.partnet_mobility_utils.articulate import articulate_joint

from non_rigid.utils.pointcloud_utils import downsample_pcd

from tqdm import tqdm


DEMOS_PER_LINK = 10
NUM_POINTS = 2048
PM_DIR = Path(os.path.expanduser("~/datasets/partnet-mobility/dataset"))
DATA_DIR = Path(os.path.expanduser("~/data/partnet/"))
OBJ_CATS = ["Microwave", "Oven", "Refrigerator", "Safe", "Dishwasher","StorageFurniture", "Table"]
LINK_LABELS = ["door", "rotation_door"]

def gen_split_demos(split_objs, split_name):
    split_dir = DATA_DIR / split_name
    if os.path.exists(split_dir):
        print("Split already exists, delete first.")
        quit()
    os.makedirs(split_dir, exist_ok=True)

    for obj in tqdm(split_objs):
        # Skip objects that are not in relevant categories.
        if obj[1] not in OBJ_CATS:
            continue
        
        # Create object, and check if it has relevant links.
        pm_obj = PMObject(PM_DIR / obj[0])
        renderer = PybulletRenderer()
        links = [js.name for js in pm_obj.semantics.by_type("hinge") if js.label in LINK_LABELS]
        if len(links) == 0:
            continue

        # Create articulation demo for each valid link.
        for i, link in enumerate(links):
            for j in range(DEMOS_PER_LINK):
                demo_name = f"{obj[0]}_{i}_{j}"
                render = renderer.render(pm_obj, camera_xyz="random")
                pos_init = render["pos"]
                t_wc = render["T_world_cam"]
                t_wb = render["T_world_base"]
                seg = render["seg"]
                angles = render["angles"]
                labelmap = render["labelmap"]

                # FPS downsampling.
                pos_init, pos_indices = downsample_pcd(
                    torch.as_tensor(pos_init).unsqueeze(0).to("cuda:1"),
                    NUM_POINTS,
                    "fps",
                )
                pos_init = pos_init.squeeze(0).cpu().numpy()
                pos_indices = pos_indices.squeeze(0).cpu().numpy()
                seg = seg[pos_indices]

                # Articulate.
                pos_new = articulate_joint(
                    obj=pm_obj,
                    current_jas=angles,
                    link_to_actuate=link,
                    amount_to_actuate=np.pi / 2,
                    pos=pos_init,
                    seg=seg,
                    labelmap=labelmap,
                    T_world_base=t_wb,
                )

                demo = {
                    "pc_init": pos_init,
                    "pc_goal": pos_new,
                    "seg": seg,
                    "obj_id": obj[0],
                    "obj_cat": obj[1],
                    "link": link,
                    "angle_init": 0,
                    "angle_goal": np.pi / 2,
                    "t_wc": t_wc,
                    "t_wb": t_wb,
                }
                np.savez(
                    split_dir / f"{demo_name}.npz",
                    **demo,
                )

if __name__ == "__main__":
    gen_split_demos(val_ood_objs, "val_ood")