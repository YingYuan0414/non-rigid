import os
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from non_rigid.utils.pointcloud_utils import downsample_pcd
from typing import Dict, List

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess RPDiff dataset")
    parser.add_argument("--data-dir", type=str, default="/data/lyuxing/tax3d/rpdiff/data/task_demos", help="Path to RPDiff data directory")
    parser.add_argument("--task-name", type=str, required=True, help="RPDiff task name")
    parser.add_argument("--task-type", type=str, required=True, help="RPDiff task type")
    parser.add_argument("--sample-size-action", type=int, default=512, help="Number of points to sample from action point cloud")
    parser.add_argument("--sample-size-anchor", type=int, default=512, help="Number of points to sample from anchor point cloud")
    parser.add_argument("--downsample-type", type=str, default="fps", choices=["fps", "random"], help="Method for downsampling points")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for preprocessed data")
    parser.add_argument("--split", type=str, default="all", choices=["train", "val", "test", "all"], help="Which split to preprocess")
    return parser.parse_args()

def load_split_files(split_dir: Path, split_type: str) -> List[str]:
    if split_type == "val":
        split_file = "train_val_split.txt"
    else:
        split_file = f"{split_type}_split.txt"
    
    with open(os.path.join(split_dir, split_file), "r") as file:
        return [line.strip() for line in file]

def preprocess_demo(
    demo_path: str, 
    device: torch.device,
    sample_size_action: int,
    sample_size_anchor: int,
    downsample_type: str,
) -> Dict[str, torch.Tensor]:
    # Load the demo file
    demo = np.load(demo_path, allow_pickle=True)
    
    # Access start and final PCDs for parent and child
    parent_start_pcd = demo['multi_obj_start_pcd'].item()['parent']
    child_start_pcd = demo['multi_obj_start_pcd'].item()['child']
    parent_final_pcd = demo['multi_obj_final_pcd'].item()['parent']
    child_final_pcd = demo['multi_obj_final_pcd'].item()['child']
    
    # Convert to tensors and move to device
    action_pc = torch.as_tensor(child_start_pcd).float().to(device)
    anchor_pc = torch.as_tensor(parent_start_pcd).float().to(device)
    goal_action_pc = torch.as_tensor(child_final_pcd).float().to(device)
    goal_anchor_pc = torch.as_tensor(parent_final_pcd).float().to(device)
    
    # Downsample action point cloud
    if sample_size_action > 0 and action_pc.shape[0] > sample_size_action:
        action_pc, action_pc_indices = downsample_pcd(action_pc.unsqueeze(0), sample_size_action, type=downsample_type)
        action_pc = action_pc.squeeze(0)
        goal_action_pc = goal_action_pc[action_pc_indices.squeeze(0)]
    
    # Downsample anchor point cloud
    if sample_size_anchor > 0 and anchor_pc.shape[0] > sample_size_anchor:
        anchor_pc, anchor_pc_indices = downsample_pcd(anchor_pc.unsqueeze(0), sample_size_anchor, type=downsample_type)
        anchor_pc = anchor_pc.squeeze(0)
        goal_anchor_pc = goal_anchor_pc[anchor_pc_indices.squeeze(0)]
        
    # Return preprocessed tensors (move back to CPU for storage)
    multi_obj_start_pcd_dict = {
        "parent": anchor_pc.cpu(),
        "child": action_pc.cpu()
    }

    multi_obj_final_pcd_dict = {
        "parent": anchor_pc.cpu(),
        "child": goal_action_pc.cpu()
    }

    preprocessed = {
        "multi_obj_start_pcd": multi_obj_start_pcd_dict,
        "multi_obj_final_pcd": multi_obj_final_pcd_dict
    }
    
    return preprocessed

def main():
    args = parse_args()
    
    # Setup directories
    root = Path(args.data_dir)
    dataset_dir = root / args.task_name / args.task_type
    split_dir = dataset_dir / "split_info"
    
    # Create output directory if not specified
    if args.output_dir is None:
        output_dir = dataset_dir / "preprocessed"
    else:
        output_dir = Path(args.output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get splits to process
    splits_to_process = ["train", "val", "test"] if args.split == "all" else [args.split]
    
    for split_type in splits_to_process:
        print(f"Processing {split_type} split...")
        
        # Load split files
        demo_files = load_split_files(split_dir, split_type)
        print(f"Found {len(demo_files)} demos for {split_type} split")
        
        # Process each demo
        for demo_file in tqdm(demo_files, desc=f"Preprocessing {split_type} demos"):
            demo_path = f"{dataset_dir}/{demo_file}"
            demo_name = os.path.basename(demo_file)
            
            try:
                preprocessed_data = preprocess_demo(
                    demo_path=demo_path,
                    device=device,
                    sample_size_action=args.sample_size_action,
                    sample_size_anchor=args.sample_size_anchor,
                    downsample_type=args.downsample_type,
                )
                
                # Save preprocessed data
                output_path = output_dir / demo_name
                #torch.save(preprocessed_data, output_path)
                np.savez(output_path, **preprocessed_data)

            except Exception as e:
                print(f"Error processing {demo_path}: {e}")
                continue
            
        print(f"Completed processing {split_type} split")
    
    print("Preprocessing complete!")

if __name__ == "__main__":
    main()
    
# Examples:
# python preprocess --task-name mug_rack_easy_single --task-type task_name_mug_on_rack