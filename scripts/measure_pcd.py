import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import norm
import torch

def load_proccloth(dir, num_demos = None):
    npz_files = [f for f in os.listdir(dir) if f.endswith('.npz')]
    loaded_action = []
    loaded_anchor = []

    if num_demos:
        npz_files = npz_files[:num_demos]

    for npz_file in npz_files:
        file_path = os.path.join(dir, npz_file)
        demo = np.load(file_path, allow_pickle=True)
        loaded_action.append(demo["action_pc"])
        loaded_anchor.append(demo["anchor_pc"])
    
    return loaded_action, loaded_anchor

def load_ndf(dir, num_demos = None):
    npz_files = [f for f in os.listdir(dir) if f.endswith('_teleport_obj_points.npz')]
    loaded_action = []
    loaded_anchor = []

    if num_demos:
        npz_files = npz_files[:num_demos]

    for npz_file in npz_files:
        file_path = os.path.join(dir, npz_file)
        demo = np.load(file_path, allow_pickle=True)

        points_raw = demo["clouds"]
        classes_raw = demo["classes"]

        points_action = points_raw[classes_raw == 0]
        points_anchor = points_raw[classes_raw == 1]

        loaded_action.append(points_action)
        loaded_anchor.append(points_anchor)

    return loaded_action, loaded_anchor

def load_rpdiff(dir, num_demos=None):
    npz_files = [f for f in os.listdir(dir) if f.endswith('.npz')]
    loaded_action = []
    loaded_anchor = []
    loaded_goal_action = []

    if num_demos:
        npz_files = npz_files[:num_demos]

    for npz_file in npz_files:
        file_path = os.path.join(dir, npz_file)
        demo = np.load(file_path, allow_pickle=True)
        points_anchor = demo['multi_obj_start_pcd'].item()['parent']
        points_action = demo['multi_obj_start_pcd'].item()['child']
        points_goal_action = demo['multi_obj_final_pcd'].item()['child']

        if points_goal_action.shape[0] < 256 or points_action.shape[0] < 256 or points_anchor.shape[0] < 512:
            print(f"[WARN] Skipping file {npz_file} due to small point cloud:")
            print(f"       goal_pcd shape:   {points_goal_action.shape}  (size={points_goal_action.size})")
            print(f"       action_pcd shape: {points_action.shape}  (size={points_action.size})")
            print(f"       anchor_pcd shape: {points_anchor.shape}  (size={points_anchor.size})")
            continue

        loaded_action.append(points_action)
        loaded_anchor.append(points_anchor)
        loaded_goal_action.append(points_goal_action)

    # Compute average point counts
    if loaded_action:
        avg_action = np.mean([pc.shape[0] for pc in loaded_action])
        avg_anchor = np.mean([pc.shape[0] for pc in loaded_anchor])
        avg_goal = np.mean([pc.shape[0] for pc in loaded_goal_action])

        print("\n[INFO] Average number of points:")
        print(f"       Action PCD      : {avg_action:.2f}")
        print(f"       Anchor PCD      : {avg_anchor:.2f}")
        print(f"       Goal Action PCD : {avg_goal:.2f}")
    else:
        print("\n[INFO] No valid demos loaded.")
    
    return loaded_action, loaded_anchor, loaded_goal_action

def calculate_mean_bounding_box_excluding_outliers(pcd_list, min_side_length=np.float32(1e-5), scaling_factor=1):
    bbox_sizes = []
    bbox_volumes = []

    # Calculate bounding box sizes and volumes for all point clouds
    for i, pcd in tqdm(enumerate(pcd_list), total=len(pcd_list), desc="Processing PCDs"):
        if pcd.shape[1] != 3:
            raise ValueError(f"Point cloud {i} must have shape [N, 3]. Got shape {pcd.shape}.")
        
        # Scale the point cloud
        scaled_pcd = pcd * scaling_factor

        # Compute axis-aligned bounding box (AABB)
        min_bound = np.min(scaled_pcd, axis=0)  # Minimum x, y, z values
        max_bound = np.max(scaled_pcd, axis=0)  # Maximum x, y, z values

        # Bounding box size and volume
        bbox_size = max_bound - min_bound

        # In case that one side of the box is too small
        adjusted_bbox_size = np.where(bbox_size < min_side_length, np.float32(1.0), bbox_size).astype(np.float32)
        bbox_volume = np.prod(adjusted_bbox_size)

        # Collect results
        bbox_sizes.append(bbox_size)
        bbox_volumes.append(bbox_volume)

    # Convert to numpy arrays for filtering
    bbox_sizes = np.array(bbox_sizes)  # Shape: [N, 3]
    bbox_volumes = np.array(bbox_volumes)  # Shape: [N]

    # Exclude outliers using the IQR method
    def exclude_outliers(data):
        Q1 = np.percentile(data, 25, axis=0)  # 25th percentile (lower quartile)
        Q3 = np.percentile(data, 75, axis=0)  # 75th percentile (upper quartile)
        IQR = Q3 - Q1  # Interquartile range
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (data >= lower_bound) & (data <= upper_bound)

    # Identify valid bounding box sizes and volumes
    valid_size_mask = exclude_outliers(bbox_sizes).all(axis=1)
    valid_volume_mask = exclude_outliers(bbox_volumes)

    # Combine masks
    valid_mask = valid_size_mask & valid_volume_mask

    # Filter valid sizes and volumes
    valid_bbox_sizes = bbox_sizes[valid_mask]
    valid_bbox_volumes = bbox_volumes[valid_mask]

    # Compute mean bounding box size and volume from valid data
    mean_bbox_size = np.mean(valid_bbox_sizes, axis=0)
    mean_bbox_volume = np.mean(valid_bbox_volumes)

    return mean_bbox_size, mean_bbox_volume, valid_mask

def compute_pcd_mean_distance(action_pcds, anchor_pcds, scaling_factor=1.0):
    """
    Compute the Euclidean distance (L2 norm) between the mean points of action and anchor point clouds.

    Args:
        action_pcds (list of np.ndarray): List of action point clouds.
        anchor_pcds (list of np.ndarray): List of anchor point clouds.
        scaling_factor (float): Factor to scale the PCD before computing the mean.

    Returns:
        np.ndarray: Array of distances between action and anchor mean points.
    """
    distances = []
    
    for action_pcd, anchor_pcd in tqdm(zip(action_pcds, anchor_pcds), total=len(action_pcds), desc="Computing distances"):
        # Scale point clouds
        action_pcd = action_pcd * scaling_factor
        anchor_pcd = anchor_pcd * scaling_factor

        # Compute mean point for each PCD
        action_mean = np.mean(action_pcd, axis=0)
        anchor_mean = np.mean(anchor_pcd, axis=0)

        # Compute Euclidean distance between mean points
        distance = np.linalg.norm(action_mean - anchor_mean)
        distances.append(distance)

    return np.array(distances)


def compute_and_plot_pcd_mean_distance(action_pcds, anchor_pcds, scaling_factor=1.0, save_dir="./vis/measure_pcd/", overlay_std=1.0):
    """
    Compute the mean point distance between action and anchor PCDs and plot the distribution 
    overlaid with a zero-centered Gaussian curve. Additionally, highlight 1σ (66.67%) and 2σ (96%) 
    regions only for values ≥ 0.

    Args:
        action_pcds (list of np.ndarray): List of action point clouds.
        anchor_pcds (list of np.ndarray): List of anchor point clouds.
        scaling_factor (float): Factor to scale the PCD before computing distances.
        save_dir (str): Directory to save the plots.
        overlay_std (float): Standard deviation for the Gaussian overlay.

    Returns:
        np.ndarray: Array of mean point distances.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Compute distances
    distances = compute_pcd_mean_distance(action_pcds, anchor_pcds, scaling_factor)

    # Compute statistics
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)

    # Define histogram bins
    bins = 30
    plt.figure(figsize=(8, 5))
    counts, bin_edges, _ = plt.hist(distances, bins=bins, color='blue', alpha=0.7, edgecolor='black', label="Empirical Data")

    # Find the maximum frequency (highest bin count)
    max_freq = max(counts)

    # Overlay Gaussian curve (zero-centered)
    x_vals = np.linspace(min(distances), max(distances), 100)
    gaussian_curve = norm.pdf(x_vals, loc=0, scale=overlay_std)  # Zero-centered Gaussian

    # Scale Gaussian to match histogram peak
    scale_factor = max_freq / max(gaussian_curve)
    plt.plot(x_vals, gaussian_curve * scale_factor, color='red', linestyle="dashed", label=f"Gaussian (std={overlay_std})")

    # Highlight only positive 1σ and 2σ regions
    max_distance = max(distances)
    if overlay_std > 0:
        plt.axvspan(0, min(overlay_std, max_distance), color='orange', alpha=0.3, label="1σ (66.67%)")
        plt.axvspan(0, min(2 * overlay_std, max_distance), color='yellow', alpha=0.2, label="2σ (96%)")

    # Labels and legend
    plt.xlabel("L2 Distance between Action and Anchor Mean Points")
    plt.ylabel("Frequency")
    plt.title("Distribution of Mean Point Distances with Gaussian Overlay")
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig(os.path.join(save_dir, f"mean_distance_hist_with_gaussian_std={overlay_std}.jpg"))
    plt.close()

    # Print summary statistics
    print(f"Mean Distance: {mean_distance:.4f}")
    print(f"Median Distance: {np.median(distances):.4f}")
    print(f"Max Distance: {np.max(distances):.4f}")
    print(f"Min Distance: {np.min(distances):.4f}")

    return distances

'''
def compute_scaling_factor(goal_action_pcds, action_pcds, anchor_pcds, mode='anchor_centroid', noisy_std=1.0, test_scale=15):
    """
    Args:
        goal_action_pcds (list of np.ndarray): List of (N, 3) goal point clouds
        anchor_pcds (list of np.ndarray): List of (M, 3) anchor point clouds
        mode (str): 'anchor centroid' or 'noisy_goal'
        noisy_std (float): std of noise if using 'noisy_goal' mode

    Returns:
        scalar_scaling (float): scalar to scale original data
        original_stats (dict): original variances and per-axis scaling
        scaled_stats (dict): variances after applying scalar scaling to original point clouds
    """
    assert mode in ['anchor_centroid', 'noisy_goal'], "Invalid mode"

    centers = []
    shapes = []
    flows = []

    print("[INFO] Calculating original variances...")
    for goal_pcd, action_pcd, anchor_pcd in tqdm(zip(goal_action_pcds, action_pcds, anchor_pcds), total=len(goal_action_pcds)):
        goal_pcd = goal_pcd.astype(np.float32)
        anchor_pcd = anchor_pcd.astype(np.float32)

        center_goal = goal_pcd.mean(axis=0)
        center_action = action_pcd.mean(axis=0)

        if mode == 'anchor_centroid':
            center_anchor = anchor_pcd.mean(axis=0)
            center = center_goal - center_anchor
            flow = (goal_pcd - center_anchor) - (action_pcd - center_action)
        elif mode == 'noisy_goal':
            center = np.random.randn(*center_goal.shape) * noisy_std
            flow = (goal_pcd - center) - (action_pcd - center_action)

        centers.append(center)
        shapes.append(goal_pcd - center_goal)
        flows.append(flow)

    centers = np.stack(centers, axis=0)
    shapes = np.concatenate(shapes, axis=0)
    flows = np.stack(flows, axis=0)

    var_R = np.var(centers, axis=0, ddof=1)
    var_S = np.var(shapes, axis=0, ddof=1)
    var_F = np.var(flows, axis=0, ddof=1)

    total_var = var_R + var_S
    axis_scaling = np.sqrt(1.0 / total_var)
    scalar_scaling = axis_scaling.mean()

    print(f"[INFO] Original center variance (R): {var_R}")
    print(f"[INFO] Original shape variance  (S): {var_S}")
    print(f"[INFO] Original flow variance  (S): {var_F}")
    print(f"[INFO] Total variance (R + S):        {total_var}")
    print(f"[INFO] Per-axis scaling factor:       {axis_scaling}")
    print(f"[INFO] Scalar scaling factor:         {scalar_scaling}")

    original_stats = {
        'var_R': var_R,
        'var_S': var_S,
        'total_var': total_var,
        'axis_scaling': axis_scaling,
        'scalar_scaling': scalar_scaling,
    }

    # === Apply scalar scaling to original data and recompute variances ===
    centers_scaled = []
    shapes_scaled = []
    flows_scaled = []

    print("[INFO] Recomputing variances after applying suggested scalar scaling...")
    for goal_pcd, action_pcd, anchor_pcd in tqdm(zip(goal_action_pcds, action_pcds, anchor_pcds), total=len(goal_action_pcds)):
        goal_pcd = goal_pcd.astype(np.float32)
        anchor_pcd = anchor_pcd.astype(np.float32)

        goal_scaled = goal_pcd * scalar_scaling
        anchor_scaled = anchor_pcd * scalar_scaling
        action_scaled = action_pcd * scalar_scaling

        center_goal = goal_scaled.mean(axis=0)
        center_action = action_scaled.mean(axis=0)

        if mode == 'anchor_centroid':
            center_anchor = anchor_pcd.mean(axis=0)
            center = center_goal - center_anchor
            flow = (goal_pcd - center_anchor) - (action_pcd - center_action)
        elif mode == 'noisy_goal':
            center = np.random.randn(*center_goal.shape) * noisy_std
            flow = (goal_pcd - center) - (action_pcd - center_action)

        centers_scaled.append(center)
        shapes_scaled.append(goal_pcd - center_goal)
        flows_scaled.append(flow)

    centers_scaled = np.stack(centers_scaled, axis=0)
    shapes_scaled = np.concatenate(shapes_scaled, axis=0)
    flows_scaled = np.stack(flows_scaled, axis=0)

    var_R_scaled = np.var(centers_scaled, axis=0, ddof=1)
    var_S_scaled = np.var(shapes_scaled, axis=0, ddof=1)
    var_F_scaled = np.var(flows_scaled, axis=0, ddof=1)
    total_var_scaled = var_R_scaled + var_S_scaled
    
    print(f"[INFO] Scaled center variance (R): {var_R_scaled}")
    print(f"[INFO] Scaled shape variance  (S): {var_S_scaled}")
    print(f"[INFO] Scaled flow variance  (S): {var_F_scaled}")
    print(f"[INFO] Total scaled variance     : {total_var_scaled}")

    print(f"[INFO] Suggested center noise std:  {np.sqrt(var_R_scaled.mean())}")
    print(f"[INFO] Suggested shape noise std :  {np.sqrt(var_S_scaled.mean())}")

    # === Apply test scalar scaling to original data and recompute variances ===
    centers_scaled = []
    shapes_scaled = []

    print("[INFO] Recomputing variances after applying test scalar scaling...")
    for goal_pcd, anchor_pcd in tqdm(zip(goal_action_pcds, anchor_pcds), total=len(goal_action_pcds)):
        goal_pcd = goal_pcd.astype(np.float32)
        anchor_pcd = anchor_pcd.astype(np.float32)

        goal_scaled = goal_pcd * test_scale
        anchor_scaled = anchor_pcd * test_scale
        action_scaled = action_pcd * test_scale

        center_goal = goal_scaled.mean(axis=0)
        center_action = action_scaled.mean(axis=0)

        if mode == 'anchor_centroid':
            center_anchor = anchor_pcd.mean(axis=0)
            center = center_goal - center_anchor
            flow = (goal_pcd - center_anchor) - (action_pcd - center_action)
        elif mode == 'noisy_goal':
            center = np.random.randn(*center_goal.shape) * noisy_std
            flow = (goal_pcd - center) - (action_pcd - center_action)

        centers_scaled.append(center)
        shapes_scaled.append(goal_pcd - center_goal)
        flows_scaled.append(flow)

    centers_scaled = np.stack(centers_scaled, axis=0)
    shapes_scaled = np.concatenate(shapes_scaled, axis=0)
    flows_scaled = np.stack(flows_scaled, axis=0)

    var_R_scaled = np.var(centers_scaled, axis=0, ddof=1)
    var_S_scaled = np.var(shapes_scaled, axis=0, ddof=1)
    var_F_scaled = np.var(flows_scaled, axis=0, ddof=1)
    total_var_scaled = var_R_scaled + var_S_scaled
    
    print(f"[INFO] TEST Scaled center variance (R): {var_R_scaled}")
    print(f"[INFO] TEST Scaled shape variance  (S): {var_S_scaled}")
    print(f"[INFO] TEST Scaled flow variance  (S): {var_F_scaled}")
    print(f"[INFO] TEST Total scaled variance     : {total_var_scaled}")
'''

import numpy as np

def compute_variance_stats(goal_pcds, action_pcds, anchor_pcds, scalar=1.0, mode='anchor_centroid', noisy_std=1.0, label=''):
    centers, shapes, flows = [], [], []

    for goal_pcd, action_pcd, anchor_pcd in zip(goal_pcds, action_pcds, anchor_pcds):
        goal_pcd = goal_pcd.astype(np.float32) * scalar
        action_pcd = action_pcd.astype(np.float32) * scalar
        anchor_pcd = anchor_pcd.astype(np.float32) * scalar

        center_goal = goal_pcd.mean(axis=0)
        center_action = action_pcd.mean(axis=0)

        if mode == 'anchor_centroid':
            center_anchor = anchor_pcd.mean(axis=0)
            center = center_goal - center_anchor
            flow = (goal_pcd - center_anchor) - (action_pcd - center_action)
        elif mode == 'noisy_goal':
            center = np.random.randn(*center_goal.shape) * noisy_std
            flow = (goal_pcd - center) - (action_pcd - center_action)

        centers.append(center)
        # Only use action object for shape variance now
        shapes.append(action_pcd - center_action)
        flows.append(flow)

    centers = np.stack(centers, axis=0)
    shapes = np.concatenate(shapes, axis=0)
    flows = np.concatenate(flows, axis=0)

    var_R = np.var(centers, axis=0, ddof=1)
    var_S = np.var(shapes, axis=0, ddof=1)
    var_F = np.var(flows, axis=0, ddof=1)
    mean_F = np.mean(flows, axis=0)
    var_F_centered = np.var(flows - mean_F, axis=0, ddof=1)

    print(f"\n[INFO] {label} center variance (R):      {var_R}")
    print(f"[INFO] {label} shape variance  (S):      {var_S}")
    print(f"[INFO] {label} flow variance   (F):      {var_F}")
    print(f"[INFO] {label} mean flow:                {mean_F}")
    print(f"[INFO] {label} flow - mean(flow) var:    {var_F_centered}")
    print(f"[INFO] {label} suggested noise stds     : center = {np.sqrt(var_R.mean()):.4f}, shape = {np.sqrt(var_S.mean()):.4f}")

    return {
        'var_R': var_R,
        'var_S': var_S,
        'var_F': var_F,
        'mean_F': mean_F,
        'var_F_centered': var_F_centered,
    }


def compute_scaling_factor(goal_action_pcds, action_pcds, anchor_pcds, mode='anchor_centroid', noisy_std=1.0, test_scale=15):
    """
    Computes a scalar scaling factor based only on the shape variance of the action object point cloud.
    """
    print("[INFO] Calculating shape-based scale from action point clouds...")
    original_stats = compute_variance_stats(
        goal_action_pcds, action_pcds, anchor_pcds,
        scalar=1.0, mode=mode, noisy_std=noisy_std, label='Original'
    )

    # Use only the shape variance (action object size)
    scalar_scaling = 1.0 / np.sqrt(original_stats['var_S'].mean())
    print(f"[INFO] Suggested scalar scaling factor (based on action shape): {scalar_scaling:.4f}")

    print("\n[INFO] Recomputing SCALED variances (suggested scale)...")
    scaled_stats = compute_variance_stats(
        goal_action_pcds, action_pcds, anchor_pcds,
        scalar=scalar_scaling, mode=mode, noisy_std=noisy_std, label='Scaled'
    )

    print("\n[INFO] Recomputing TEST SCALED variances (test scale)...")
    test_scaled_stats = compute_variance_stats(
        goal_action_pcds, action_pcds, anchor_pcds,
        scalar=test_scale, mode=mode, noisy_std=noisy_std, label='TestScaled'
    )

    return scalar_scaling, original_stats, scaled_stats, test_scaled_stats


def compute_extent_and_scale_stats(pcd_list):
    """
    Compute statistics of point cloud extents and scaling factors.

    Args:
        pcd_list (List[np.ndarray or torch.Tensor]): 
            List of point clouds of shape (N_i, 3)

    Returns:
        dict: Dictionary with:
            - 'mean_extent': np.ndarray of shape (3,)
            - 'min_extent': np.ndarray of shape (3,)
            - 'max_extent': np.ndarray of shape (3,)
            - 'extent_25': np.ndarray of shape (3,)
            - 'extent_75': np.ndarray of shape (3,)
            - 'mean_scale': float
            - 'scale_25': float
            - 'scale_75': float
    """
    extents = []
    scales = []

    for pc in pcd_list:
        if isinstance(pc, torch.Tensor):
            pc = pc.detach().cpu().numpy()

        center = pc.mean(axis=0)
        pc_centered = pc - center

        min_xyz = pc_centered.min(axis=0)
        max_xyz = pc_centered.max(axis=0)
        extent = max_xyz - min_xyz
        avg_extent = extent.mean()
        scale = 1.0 / avg_extent

        extents.append(extent)
        scales.append(scale)

    extents = np.stack(extents, axis=0)
    scales = np.array(scales)

    return {
        "mean_extent": extents.mean(axis=0),
        "min_extent": extents.min(axis=0),
        "max_extent": extents.max(axis=0),
        "extent_25": np.percentile(extents, 25, axis=0),
        "extent_75": np.percentile(extents, 75, axis=0),
        "mean_scale": scales.mean(),
        "scale_25": np.percentile(scales, 25),
        "scale_75": np.percentile(scales, 75)
    }


def print_extent_stats(name, stats):
    """
    Nicely print extent and scale stats.

    Args:
        name (str): Label for the point cloud group
        stats (dict): Output from compute_extent_and_scale_stats
    """
    print(f"\n=== Stats for {name} ===")
    print(f"Mean Extent (x, y, z): {stats['mean_extent']}")
    print(f"Min Extent  (x, y, z): {stats['min_extent']}")
    print(f"Max Extent  (x, y, z): {stats['max_extent']}")
    print(f"25% Extent (x, y, z):  {stats['extent_25']}")
    print(f"75% Extent (x, y, z):  {stats['extent_75']}")
    print(f"Mean Scale:            {stats['mean_scale']:.5f}")
    print(f"25% Scale:             {stats['scale_25']:.5f}")
    print(f"75% Scale:             {stats['scale_75']:.5f}")

# Example usage
if __name__ == "__main__":
    # Create a list of example point clouds
    '''
    proccloth_action, proccloth_anchor = load_proccloth(dir='/data/lyuxing/tax3d/proccloth/cloth=multi-fixed anchor=single-random hole=single/train_tax3d/')

    mean_bbox_size, mean_bbox_volume, valid_mask = calculate_mean_bounding_box_excluding_outliers(proccloth_action, scaling_factor=1)
    print(f"ProcCloth Action Mean Bounding Box Size (Excluding Outliers): {mean_bbox_size}")
    print(f"ProcCloth Action Mean Bounding Box Volume (Excluding Outliers): {mean_bbox_volume:.2f}")

    mean_bbox_size, mean_bbox_volume, valid_mask = calculate_mean_bounding_box_excluding_outliers(proccloth_anchor, scaling_factor=1)
    print(f"ProcCloth Anchor Mean Bounding Box Size (Excluding Outliers): {mean_bbox_size}")
    print(f"ProcCloth Anchor Bounding Box Volume (Excluding Outliers): {mean_bbox_volume:.2f}")



    ndf_action, ndf_anchor = load_ndf(dir='/data/lyuxing/tax3d/ndf/mugplace/train_data/renders/')

    mean_bbox_size, mean_bbox_volume, valid_mask = calculate_mean_bounding_box_excluding_outliers(ndf_action, scaling_factor=15)
    print(f"NDF Action Mean Bounding Box Size (Excluding Outliers): {mean_bbox_size}")
    print(f"NDF Action Mean Bounding Box Volume (Excluding Outliers): {mean_bbox_volume:.2f}")

    mean_bbox_size, mean_bbox_volume, valid_mask = calculate_mean_bounding_box_excluding_outliers(ndf_anchor, scaling_factor=15)
    print(f"NDF Anchor Mean Bounding Box Size (Excluding Outliers): {mean_bbox_size}")
    print(f"NDF Anchor Bounding Box Volume (Excluding Outliers): {mean_bbox_volume:.2f}")
    '''
    
    #rpdiff_action, rpdiff_anchor, rpdiff_goal_action = load_rpdiff(dir='/data/lyuxing/tax3d/rpdiff/data/task_demos/can_in_cabinet_stack/task_name_stack_can_in_cabinet/', num_demos=None)
    #rpdiff_action, rpdiff_anchor, rpdiff_goal_action = load_rpdiff(dir='/data/lyuxing/tax3d/rpdiff/data/task_demos/book_on_bookshelf_double_view_rnd_ori/task_name_book_in_bookshelf/preprocessed/', num_demos=None)
    #rpdiff_action, rpdiff_anchor, rpdiff_goal_action = load_rpdiff(dir='/data/lyuxing/tax3d/rpdiff/data/task_demos/mug_rack_easy_single//task_name_mug_on_rack/preprocessed/', num_demos=None)
    #rpdiff_action, rpdiff_anchor, rpdiff_goal_action = load_rpdiff(dir='/data/lyuxing/tax3d/rpdiff/data/task_demos/mug_rack_med_single/task_name_mug_on_rack/preprocessed/', num_demos=None)
    #rpdiff_action, rpdiff_anchor, rpdiff_goal_action = load_rpdiff(dir='/data/lyuxing/tax3d/rpdiff/data/task_demos/mug_on_rack_multi_large_proc_gen_demos/task_name_mug_on_rack_multi/preprocessed/', num_demos=None)
    #rpdiff_action, rpdiff_anchor, rpdiff_goal_action = load_rpdiff(dir='/data/lyuxing/tax3d/rpdiff/data/task_demos/mug_rack_med_multi/task_name_mug_on_rack_multi/preprocessed/', num_demos=None)

    dedo_action, dedo_anchor = load_proccloth('/home/lyuxing/Desktop/tax3d_upgrade/datasets/ProcCloth/cloth=single-fixed anchor=single-random hole=single/train_tax3d')
    
    stats_action = compute_extent_and_scale_stats(dedo_action)
    stats_anchor = compute_extent_and_scale_stats(dedo_anchor)
    stats_combined = compute_extent_and_scale_stats(
        [np.concatenate((pc1, pc2), axis=0) for pc1, pc2 in zip(dedo_action, dedo_anchor)]
    )
    
    '''
    test_scale = 15
    rpdiff_action = [pc * test_scale for pc in rpdiff_action]
    rpdiff_anchor = [pc * test_scale for pc in rpdiff_anchor]

    stats_action = compute_extent_and_scale_stats(rpdiff_action)
    stats_anchor = compute_extent_and_scale_stats(rpdiff_anchor)
    stats_combined = compute_extent_and_scale_stats(
        [np.concatenate((pc1, pc2), axis=0) for pc1, pc2 in zip(rpdiff_action, rpdiff_anchor)]
    )
    '''
    
    print_extent_stats("Action", stats_action)
    print_extent_stats("Anchor", stats_anchor)
    print_extent_stats("Combined", stats_combined)