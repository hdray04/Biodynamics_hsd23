import ezc3d
import numpy as np 
import matplotlib as plt 
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


cmj_1 = ezc3d.c3d("/Users/harrietdray/Biodynamics/Harriet_c3d/CMJ-001/pose_filt_0.c3d")
cmj_2 = ezc3d.c3d("/Users/harrietdray/Biodynamics/Harriet_c3d/CMJ-002/pose_filt_0.c3d")
cmj_3 = ezc3d.c3d("/Users/harrietdray/Biodynamics/Harriet_c3d/CMJ-003/pose_filt_0.c3d")

angle_data_trials = [cmj_1['data']['points'], cmj_2['data']['points'], cmj_3['data']['points']]
rotation_data_trials = [cmj_1['data']['rotations'], cmj_2['data']['rotations'], cmj_3['data']['rotations']]
cmj_data = [cmj_1, cmj_2, cmj_3]  # List of c3d data objects
labels = cmj_data[0]['parameters']['POINT']['LABELS']['value'] # List of angle names
labels_rotation = cmj_data[0]['parameters']['ROTATION']['LABELS']['value']
units = cmj_data[0]['parameters']['POINT']['UNITS']['value']
print("Units used:", units)


def get_frame_rate(c3d_file_path):
    # Extract the frame rate from the POINT parameter block
    frame_rate = float(cmj_1['parameters']['POINT']['RATE']['value'][0])
    return frame_rate

# Example usage
frame_rate = get_frame_rate(cmj_1)
print(f"Frame rate: {frame_rate} Hz")

def extract_3d_angles(angle_data_trials, labels):
    angle_indices = {
        'left_knee': labels.index('LeftKneeAngles_Theia'),
        'right_knee': labels.index('RightKneeAngles_Theia'),
        'left_hip': labels.index('LeftHipAngles_Theia'),
        'right_hip': labels.index('RightHipAngles_Theia'),
        'left_fp': labels.index('LeftFootProgressionAngles_Theia'),
        'right_fp': labels.index('RightFootProgressionAngles_Theia'),
        'left_ankle': labels.index('LeftAnkleAngles_Theia'),
        'right_ankle': labels.index('RightAnkleAngles_Theia')

    }

    all_angles = {}
    for trial_idx, angle_data in enumerate(angle_data_trials):
        n_frames = angle_data.shape[2]
        for joint_name, joint_idx in angle_indices.items():
            # Extract sagittal, frontal, and transverse angles (assuming 3D data is stored in the first 3 rows)
            sagittal = angle_data[0, joint_idx, :n_frames]  # X-axis (sagittal plane)
            frontal = angle_data[1, joint_idx, :n_frames]   # Y-axis (frontal plane)
            transverse = angle_data[2, joint_idx, :n_frames]  # Z-axis (transverse plane)

            # Store the 3D angles for the joint
            all_angles[joint_name] = {
                'sagittal': sagittal,
                'frontal': frontal,
                'transverse': transverse
            }

        # Store the angles for this trial
        all_angles[f'trial_{trial_idx + 1}'] = all_angles

    return all_angles


# Extract 3D angles for all trials
all_angles_3d = extract_3d_angles(angle_data_trials, labels)



def extract_matrices(rotation_data_trials, labels_rotation):# Shape: (4, 4, 19, 547)
    all_matrices = {} # Dictionary to hold matrices for each joint
    
    for trial_idx, rotation_data_ in enumerate(rotation_data_trials):
        rotation_data = cmj_data[trial_idx]['data']['rotations']  # Extract rotation data for the current trial
        matrices_dict = {}
        n_joints = rotation_data.shape[2]  # 19 joints
        n_frames = rotation_data.shape[3]  # 547 frames
    
    
        for joint_idx in range(n_joints):
            if joint_idx < len(labels_rotation):
                label = labels_rotation[joint_idx]
                joint_name = label.replace('_4X4', '')
            
            # Extract all matrices for this joint across all frames
            # Shape will be (4, 4, n_frames) -> we want (n_frames, 4, 4)
                joint_matrices = rotation_data[:, :, joint_idx, :].transpose(2, 0, 1)
            
                matrices_dict[joint_name] = joint_matrices
        
        print(f"Extracted {len(matrices_dict)} joints for trial {trial_idx + 1}")
        
        # Store the matrices for this trial
        all_matrices[f'trial_{trial_idx + 1}'] = matrices_dict

    return all_matrices
    
  

def extract_positions_from_matrices(all_matrices):
    all_positions = {}

    for trial_name, matrices in all_matrices.items():
        positions_dict = {}

        for joint_name, joint_matrices in matrices.items():
            # Extract the translation (position) component from the 4x4 matrices
            # Translation is in the last column of the 4x4 matrix
            positions = joint_matrices[:, :3, 3]  # Shape: (n_frames, 3)
            positions_dict[joint_name] = positions

        all_positions[trial_name] = positions_dict

    return all_positions

positions = extract_positions_from_matrices(extract_matrices(rotation_data_trials, labels_rotation))

# Joints from your data: pelvis, torso, head,
# l_thigh, l_shank, l_foot, l_toes, r_thigh, r_shank, r_foot, r_toes,
# l_uarm, l_larm, l_hand, r_uarm, r_larm, r_hand

SEGMENTS = {
    # --- Lower limb (LEFT) ---
    # Thigh: hip (l_thigh) -> knee (l_shank)
    "thigh_L": (0.1000, 0.433, "l_thigh", "l_shank"),
    # Shank/Leg: knee (l_shank) -> ankle (l_foot)
    "shank_L": (0.0465, 0.433, "l_shank", "l_foot"),
    # Foot: ankle (l_foot) -> toes (l_toes)
    "foot_L":  (0.0145, 0.500, "l_foot",  "l_toes"),

    # --- Lower limb (RIGHT) ---
    "thigh_R": (0.1000, 0.433, "r_thigh", "r_shank"),
    "shank_R": (0.0465, 0.433, "r_shank", "r_foot"),
    "foot_R":  (0.0145, 0.500, "r_foot",  "r_toes"),

    # --- Upper limb (use Winter's "Total arm" so you only need shoulder and wrist) ---
    # Total arm: shoulder (uarm) -> wrist (hand)
    "arm_L":   (0.0500, 0.530, "l_uarm",  "l_hand"),
    "arm_R":   (0.0500, 0.530, "r_uarm",  "r_hand"),

    # --- Trunk + Head (no arms) ---
    # Trunk-Head-Neck (THN): pelvis -> head
    # Winter row "Trunk head neck": mass 0.578, COM prox frac ~0.66
    "trunk_head_neck": (0.5780, 0.660, "pelvis", "head"),
}

def compute_whole_body_com(joints, body_mass, fs, g_vec=np.array([0, 0, -9.81])):
    """
    Compute the whole-body center of mass (COM) based on segment COMs.

    Parameters:
    - joints: dict {joint_name: np.ndarray [T,3]} in the SAME global frame.
    - body_mass: float (kg)
    - fs: sampling rate (Hz)
    - g_vec: gravity vector in YOUR global frame (m/s^2)

    Returns:
    - dict with COM pos/vel/acc, and net external force.
    """
    # Extract segment data into arrays for vectorized computation
    segment_masses = []
    segment_coms = []

    for seg_name, (w_i, f_i, prox, dist) in SEGMENTS.items():
        r_prox = joints[prox]  # [T,3]
        r_dist = joints[dist]  # [T,3]
        r_seg_com = r_prox + f_i * (r_dist - r_prox)  # [T,3]
        segment_coms.append(r_seg_com)
        segment_masses.append(w_i * body_mass)

    # Convert to numpy arrays for vectorized computation
    segment_coms = np.array(segment_coms)  # Shape: [N_segments, T, 3]
    segment_masses = np.array(segment_masses)  # Shape: [N_segments]

    # Compute the whole-body COM
    total_mass = np.sum(segment_masses)
    weighted_coms = segment_coms * segment_masses[:, np.newaxis, np.newaxis]  # Broadcast masses
    whole_body_com = np.sum(weighted_coms, axis=0) / total_mass  # Shape: [T, 3]

    # Basic checks
    if not np.isclose(np.sum(segment_masses) / body_mass, 1.0, atol=1e-3):
        raise ValueError("Segment mass fractions do not sum to 1.0")

    # Return results
    return {
        "com_position": whole_body_com,  # [T, 3]
        "total_mass": total_mass,
    }


fs = 100.0        # your sampling rate (Hz)
body_mass = 54.0  # kg
g_vec = np.array([0, 0, -9.81])  # set to your lab frame (this is Z-down)

out = compute_whole_body_com(positions, body_mass, fs, g_vec=g_vec)
r_com = out["r_com"]   # [T,3]
v_com = out["v_com"]
a_com = out["a_com"]
F_ext = out["F_ext"]   # net external force estimate [N]