'''Main functions shared across the files.'''


import ezc3d


def load_data(fpath):
    cmj_1 = ezc3d.c3d(fpath)
    labels_rotation = cmj_1['parameters']['ROTATION']['LABELS']['value']
    return cmj_1, labels_rotation


def extract_matrices(rotation_data, labels_rotation):
    matrices_dict = {}
    n_joints = rotation_data['data']['rotations'].shape[2]
    for joint_idx in range(n_joints):
        if joint_idx < len(labels_rotation):
            label = labels_rotation[joint_idx].replace('_4X4', '')
            joint_matrices = rotation_data['data']['rotations'][:, :, joint_idx, :].transpose(2, 0, 1)
            matrices_dict[label] = joint_matrices
    return matrices_dict


def extract_positions_from_matrices(matrices):
    positions_dict = {}
    for joint_name, joint_matrices in matrices.items():
        positions = joint_matrices[:, :3, 3]  # (frames, 3)
        positions_dict[joint_name] = positions
    return positions_dict


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