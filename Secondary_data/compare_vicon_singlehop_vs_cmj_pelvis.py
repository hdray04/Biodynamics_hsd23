import os
import numpy as np
import matplotlib.pyplot as plt
import ezc3d


def _find_label_key(candidates, available_keys):
    low = {k.lower(): k for k in available_keys}
    for c in candidates:
        k = c.lower()
        if k in low:
            return low[k]
    # fuzzy contains-all-tokens match
    for cand in candidates:
        tokens = cand.lower().split('_')
        for k in available_keys:
            lk = k.lower()
            if all(t in lk for t in tokens):
                return k
    return None


def extract_matrices(rotation_data, labels_rotation):
    matrices_dict = {}
    n_joints = rotation_data.shape[2]
    for joint_idx in range(n_joints):
        if joint_idx < len(labels_rotation):
            label = labels_rotation[joint_idx]
            joint_name = label.replace('_4X4', '')
            joint_matrices = rotation_data[:, :, joint_idx, :].transpose(2, 0, 1)
            matrices_dict[joint_name] = joint_matrices
    return matrices_dict


def extract_positions_from_matrices(matrices_dict):
    positions_dict = {}
    for joint_name, joint_matrices in matrices_dict.items():
        # translation column of 4x4 matrix
        positions = joint_matrices[:, :3, 3]
        positions_dict[joint_name] = positions
    return positions_dict


def load_vicon_single_hop_left_hip_flexion(c3d_path):
    c = ezc3d.c3d(c3d_path)
    points = c['data']['points']
    labels = c['parameters']['POINT']['LABELS']['value']
    point_rate = float(c['parameters']['POINT']['RATE']['value'][0])
    time = np.arange(points.shape[2]) / point_rate
    # Left hip flexion X angle (deg) stored in POINT channel named 'LHipAngles'
    if 'LHipAngles' not in labels:
        raise KeyError("'LHipAngles' not found in POINT.LABELS for Vicon single hop file")
    lhip_idx = labels.index('LHipAngles')
    lhip_x = points[0, lhip_idx, :]
    return time, lhip_x


def load_cmj_pelvis_height(c3d_path, fs_override=None):
    c = ezc3d.c3d(c3d_path)
    rotation_data = c['data']['rotations']
    labels_rotation = c['parameters']['ROTATION']['LABELS']['value']
    # Time base
    if fs_override is not None:
        fs = float(fs_override)
    else:
        # fallback to POINT.RATE if available
        fs = float(c['parameters']['POINT']['RATE']['value'][0]) if 'RATE' in c['parameters']['POINT'] else 100.0
    n_frames = rotation_data.shape[-1]
    time = np.arange(n_frames) / fs
    # Extract pelvis Z from rotation matrices
    matrices = extract_matrices(rotation_data, labels_rotation)
    positions = extract_positions_from_matrices(matrices)
    pelvis_key = _find_label_key(['pelvis', 'Pelvis', 'PELVIS'], positions.keys())
    if pelvis_key is None:
        # Some datasets use segment names like 'pelvis_seg' or 'root'
        pelvis_key = _find_label_key(['pelvis_seg', 'root', 'sacrum'], positions.keys())
    if pelvis_key is None:
        raise KeyError(f"Could not find pelvis key in ROTATION matrices. Available keys: {list(positions.keys())}")
    pelvis_z = positions[pelvis_key][:, 2]
    return time, pelvis_z, pelvis_key


def main():
    # Paths copied from existing scripts in this repo
    vicon_single_hop_path = \
        "/Users/harrietdray/Baseline/Tash_single_hop_left_2.c3d"
    cmj_c3d_path = \
        "/Users/harrietdray/Library/CloudStorage/OneDrive-ImperialCollegeLondon/ACL_data/Pilot - Tash/Pilot - Tash_c3d - Sorted/single_hop_left_2/pose_filt_0.c3d"

    v_time, v_lhip = load_vicon_single_hop_left_hip_flexion(vicon_single_hop_path)
    c_time, c_pelvis_z, pelvis_key = load_cmj_pelvis_height(cmj_c3d_path, fs_override=100)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=False)

    # Left: Vicon left hip flexion
    axes[0].plot(v_time, v_lhip, label='Left Hip Flexion (X)')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Flexion Angle (deg)')
    axes[0].set_title('Vicon Single Hop – Left Hip Flexion')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Right: CMJ pelvis height
    axes[1].plot(c_time, c_pelvis_z, color='tab:orange', label=f'{pelvis_key} Z (Pelvis Height)')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Height (mm)')
    axes[1].set_title('CMJ – Pelvis Height')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle('Comparison: Vicon Left Hip Flexion vs CMJ Pelvis Height', y=1.02)
    plt.tight_layout()

    # Save figure to a new document (PNG)
    out_dir = os.path.join('figures')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'overlay_singlehop_hip_vs_cmj_pelvis.png')
    plt.savefig(out_path, dpi=200)
    print(f"Saved comparison figure to {out_path}")
    plt.show()


if __name__ == '__main__':
    main()

