"""Utilities for loading C3D data and extracting joint positions.

This module provides small helpers used across the repository to:
- Load a C3D file with `ezc3d`.
- Convert the rotation matrices stored in the file into per-joint 4x4
  transformation matrices and 3D positions (mm).

Typical usage:
    >>> from src import utils
    >>> c3d, rot_labels = utils.load_data("/path/to/file.c3d")
    >>> mats = utils.extract_matrices(c3d, rot_labels)
    >>> positions = utils.extract_positions_from_matrices(mats)

All positions are returned in millimetres to match the source data.
"""


import ezc3d
from typing import Dict, Tuple, Any


def load_data(fpath: str) -> Tuple[ezc3d.c3d, Any]:
    """Load a C3D file and return its object and rotation labels.

    Parameters:
    - fpath: Path to a `.c3d` file.

    Returns:
    - (c3d_obj, rotation_labels)
    """
    cmj_1 = ezc3d.c3d(fpath)
    labels_rotation = cmj_1['parameters']['ROTATION']['LABELS']['value']
    return cmj_1, labels_rotation


def extract_matrices(rotation_data: ezc3d.c3d, labels_rotation) -> Dict[str, Any]:
    """Build a dict of joint 4x4 matrices from C3D rotation data.

    The C3D contains a 4x4 transform per joint per frame. This function
    returns a mapping of joint label -> array of shape (frames, 4, 4).
    """
    matrices_dict: Dict[str, Any] = {}
    n_joints = rotation_data['data']['rotations'].shape[2]
    for joint_idx in range(n_joints):
        if joint_idx < len(labels_rotation):
            label = labels_rotation[joint_idx].replace('_4X4', '')
            # Original layout is (4, 4, joint, frames); transpose to (frames, 4, 4)
            joint_matrices = rotation_data['data']['rotations'][:, :, joint_idx, :].transpose(2, 0, 1)
            matrices_dict[label] = joint_matrices
    return matrices_dict


def extract_positions_from_matrices(matrices: Dict[str, Any]) -> Dict[str, Any]:
    """Extract 3D positions (x, y, z) from 4x4 matrices per joint.

    Parameters:
    - matrices: dict of joint -> (frames, 4, 4)

    Returns:
    - dict of joint -> (frames, 3) positions in mm
    """
    positions_dict: Dict[str, Any] = {}
    for joint_name, joint_matrices in matrices.items():
        positions = joint_matrices[:, :3, 3]  # (frames, 3)
        positions_dict[joint_name] = positions
    return positions_dict


# Segment mass fractions and COM proportions from biomechanics literature.
# Each entry is: (mass_fraction, com_proportion_from_proximal, proximal_label, distal_label)
SEGMENTS = { # from literature
    "thigh_L": (0.1000, 0.433, "l_thigh", "l_shank"),
    "shank_L": (0.0465, 0.433, "l_shank", "l_foot"),
    "foot_L":  (0.0145, 0.500, "l_foot",  "l_toes"),
    "thigh_R": (0.1000, 0.433, "r_thigh", "r_shank"),
    "shank_R": (0.0465, 0.433, "r_shank", "r_foot"),
    "foot_R":  (0.0145, 0.500, "r_foot",  "r_toes"),
    "arm_L":   (0.0500, 0.530, "l_uarm",  "l_hand"),
    "arm_R":   (0.0500, 0.530, "r_uarm",  "r_hand"),
    "trunk_head_neck": (0.5780, 0.660, "pelvis", "head"),
}
