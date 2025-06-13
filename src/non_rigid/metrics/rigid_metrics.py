import numpy as np
import torch
from typing import Union, Tuple


def svd_estimation(
    source: torch.Tensor,
    target: torch.Tensor,
    return_magnitude: bool = False,
):
    """
    Estimate translation and rotation using SVD-based Procrustes analysis for batched inputs,
    with an option to return the magnitudes of errors.

    Args:
        source (torch.Tensor): Source point clouds, shape (B * num_samples, N, C).
        target (torch.Tensor): Target point clouds, shape (B * num_samples, N, C).
        return_magnitude (bool): If True, returns the norm of translation and the
                                  rotation angle (in degrees) instead of raw values.

    Returns:
        If return_magnitude is False:
            estimated_translations (torch.Tensor): Estimated translations, shape (B * num_samples, C).
            estimated_rotations (torch.Tensor): Estimated rotation matrices, shape (B * num_samples, C, C).
        If return_magnitude is True:
            translation_errors (torch.Tensor): Norms of the translation vectors, shape (B * num_samples,).
            rotation_errors (torch.Tensor): Rotation angles in degrees, shape (B * num_samples,).
    """
    B, N, C = source.shape
    assert target.shape == source.shape, "Source and target must have the same shape"

    # Compute centroids
    source_centroid = source.mean(dim=1)  # Shape (B, C)
    target_centroid = target.mean(dim=1)  # Shape (B, C)

    # Center the point clouds
    source_centered = source - source_centroid[:, None, :]  # Shape (B, N, C)
    target_centered = target - target_centroid[:, None, :]  # Shape (B, N, C)

    # Compute the covariance matrix
    H = torch.einsum("bij,bik->bjk", source_centered, target_centered)  # Shape (B, C, C)

    # Singular Value Decomposition
    U, _, Vt = torch.linalg.svd(H)  # U: (B, C, C), Vt: (B, C, C)

    # Compute the rotation matrices
    R = torch.bmm(Vt.transpose(1, 2), U.transpose(1, 2))  # Shape (B, C, C)

    # Ensure proper rotation (det(R) = 1, not -1)
    det_check = torch.det(R) < 0
    if det_check.any():
        Vt[det_check, -1, :] *= -1
        R = torch.bmm(Vt.transpose(1, 2), U.transpose(1, 2))

    if return_magnitude:
        # Compute translation norms. Note that here we are estimating the centroid difference!
        estimated_translations = target_centroid - source_centroid  # Shape (B, C)
        translation_errors = torch.norm(estimated_translations, dim=1)  # Shape (B,)

        # Compute rotation angles in degrees
        traces = torch.einsum("bii->b", R)  # Trace of each rotation matrix
        clamped_cos = torch.clamp((traces - 1) / 2, min=-1.0, max=1.0)
        rotation_angles = torch.acos(clamped_cos)  # In radians
        rotation_errors = torch.rad2deg(rotation_angles)     # Shape (B,)

        return translation_errors, rotation_errors
    else:
        t = target_centroid - torch.bmm(R, source_centroid.unsqueeze(-1)).squeeze(-1)  # (B, C)

        # Construct full SE(3) matrix: (B, 4, 4)
        T = torch.eye(4, device=source.device).unsqueeze(0).repeat(B, 1, 1)  # (B, 4, 4)
        T[:, :3, :3] = R
        T[:, :3, 3] = t

        return T



def translation_err(gt_translation: torch.Tensor,
                    estimated_translation: torch.Tensor,):
    """
    Calculate translation errors.

    Args:
        gt_translation (torch.Tensor): Ground truth translation, shape (3,).
        estimated_translation (torch.Tensor): Estimated translation, shape (3,).

    Returns:
        translation_error (float): Translation error (Euclidean distance).
    """
    translation_error = torch.norm(gt_translation - estimated_translation).item()
    
    return translation_error

def rotation_err(gt_rotation: torch.Tensor,
                estimated_rotation: torch.Tensor,
                metrics='degree'):
    """
    Calculate translation errors.

    Args:
        gt_rotation (torch.Tensor): Ground truth rotation matrix, shape (3, 3).
        estimated_rotation (torch.Tensor): Estimated rotation matrix, shape (3, 3).
        metrics: Error in degree or radiants

    Returns:
        rotation_error_deg (float): Rotation error in degrees or radiants.
    """
    R_diff = gt_rotation @ estimated_rotation.T
    trace = torch.trace(R_diff)  # Tr(R)
    rotation_error_rad = torch.arccos((trace - 1) / 2)
    rotation_error_deg = torch.rad2deg(rotation_error_rad).item()    

    if metrics == "degree":
        translation_error = rotation_error_deg
    elif metrics == "radian":
        translation_error = rotation_error_rad
    else:
        raise ValueError("Incorrect metrics type")

    return translation_error

'''
def calculate_errors(
    gt_translation: torch.Tensor,
    gt_rotation: torch.Tensor,
    estimated_translation: torch.Tensor,
    estimated_rotation: torch.Tensor,
):
    """
    Calculate translation and rotation errors.

    Args:
        gt_translation (torch.Tensor): Ground truth translation, shape (3,).
        gt_rotation (torch.Tensor): Ground truth rotation matrix, shape (3, 3).
        estimated_translation (torch.Tensor): Estimated translation, shape (3,).
        estimated_rotation (torch.Tensor): Estimated rotation matrix, shape (3, 3).

    Returns:
        translation_error (float): Translation error (Euclidean distance).
        rotation_error_deg (float): Rotation error in degrees.
    """
    # Translation error
    translation_error = torch.norm(gt_translation - estimated_translation).item()

    # Rotation error (angle between rotation matrices)
    R_diff = gt_rotation @ estimated_rotation.T
    trace = torch.trace(R_diff)  # Tr(R)
    rotation_error_rad = torch.arccos((trace - 1) / 2)
    rotation_error_deg = torch.rad2deg(rotation_error_rad).item()

    return translation_error, rotation_error_deg
'''