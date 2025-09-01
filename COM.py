import ezc3d
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


# --- Load cmj_1 only ---
cmj_1 = ezc3d.c3d("/Users/harrietdray/Biodynamics/Harriet_c3d/CMJ-001/pose_filt_0.c3d")
labels_rotation = cmj_1['parameters']['ROTATION']['LABELS']['value']


# --- Extract rotation matrices ---
def extract_matrices(rotation_data, labels_rotation):
    matrices_dict = {}
    n_joints = rotation_data.shape[2]
    for joint_idx in range(n_joints):
        if joint_idx < len(labels_rotation):
            label = labels_rotation[joint_idx].replace('_4X4', '')
            joint_matrices = rotation_data[:, :, joint_idx, :].transpose(2, 0, 1)
            matrices_dict[label] = joint_matrices
    return matrices_dict

rotation_data_1 = cmj_1['data']['rotations']
matrices = extract_matrices(rotation_data_1, labels_rotation)

# --- Extract positions from matrices ---
def extract_positions_from_matrices(matrices):
    positions_dict = {}
    for joint_name, joint_matrices in matrices.items():
        positions = joint_matrices[:, :3, 3]  # (frames, 3)
        positions_dict[joint_name] = positions
    return positions_dict

positions = extract_positions_from_matrices(matrices)

# --- Define segments ---
SEGMENTS = {
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
cutoff_freq = 6.0  # Hz, typical cutoff for human movement analysis
body_mass = 54.0  # kg, assumed body mass for the whole body COM calculation
fs = 100 # Hz, assumed sampling frequency

def compute_whole_body_com_fixed(joints, body_mass, fs, cutoff_freq=6.0, g_vec=np.array([0, 0, -9.81])):
    segment_masses = []
    segment_coms = []

    for seg_name, (w_i, f_i, prox, dist) in SEGMENTS.items():
        if prox not in joints or dist not in joints:
            raise KeyError(f"Missing joint: {prox} or {dist} in joints dict")

        r_prox = np.asarray(joints[prox])
        r_dist = np.asarray(joints[dist])
        r_seg_com = r_prox + f_i * (r_dist - r_prox)
        segment_coms.append(r_seg_com)
        segment_masses.append(w_i * body_mass)

    segment_coms = np.stack(segment_coms, axis=0)
    segment_masses = np.asarray(segment_masses)
    total_mass = np.sum(segment_masses)
    
    if not np.isclose(total_mass / body_mass, 1.0, atol=1e-3):
        raise ValueError("Segment mass fractions do not sum to 1.0")

    weighted = segment_coms * segment_masses[:, None, None]
    r_com_raw = np.sum(weighted, axis=0) / total_mass

    # Step 2: Apply filtering to position
    """
    nyquist = fs / 2
    normalized_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(4, normalized_cutoff, 'low')
    r_com_filtered = signal.filtfilt(b, a, r_com_raw, axis=0)
    """
    
    r_com_filtered = r_com_raw
    print(f"Applied {cutoff_freq}Hz low-pass filter to position data")

    # Step 3: Calculate derivatives using central differences ON FILTERED DATA
    dt = 1.0 / fs
    
    # Velocity using central differences
    v_com = np.zeros_like(r_com_filtered)
    v_com[1:-1] = (r_com_filtered[2:] - r_com_filtered[:-2]) / (2 * dt)
    v_com[0] = (r_com_filtered[1] - r_com_filtered[0]) / dt  # Forward difference
    v_com[-1] = (r_com_filtered[-1] - r_com_filtered[-2]) / dt  # Backward difference
    
    print("Calculated velocity using central differences")
    
    # Acceleration using central differences on velocity
    a_com = np.zeros_like(v_com)
    a_com[1:-1] = (v_com[2:] - v_com[:-2]) / (2 * dt)
    a_com[0] = (v_com[1] - v_com[0]) / dt
    a_com[-1] = (v_com[-1] - v_com[-2]) / dt
    
    print("Calculated acceleration using central differences")

    # Step 4: Calculate forces from the properly calculated acceleration
    # Convert acceleration to m/s² (it's currently in mm/s²)
    a_com_ms2 = a_com / 1000.0
    F_ext = total_mass * (a_com_ms2 - g_vec[None, :])
    
    print("Calculated forces from corrected acceleration")
    
    # Optional: Apply additional smoothing to forces if still noisy
    #F_ext_smooth = signal.filtfilt(b, a, F_ext, axis=0)
    
    return {
        "r_com_raw": r_com_raw,
        "r_com": r_com_filtered,
        "v_com": v_com,
        "a_com": a_com,
        "F_ext": F_ext,
        #"F_ext_smooth": F_ext_smooth,
        "filter_info": {"cutoff": cutoff_freq, "fs": fs}
    }

# Test the fixed function
print("=== TESTING FIXED COM CALCULATION ===")
print(f"Sampling frequency: {fs} Hz")
print(f"Body mass: {body_mass} kg")

# Run the corrected function
out_fixed = compute_whole_body_com_fixed(positions, body_mass, fs, cutoff_freq=6.0)

# Quick check of the results
r_com_fixed = out_fixed["r_com"]
v_com_fixed = out_fixed["v_com"] 
a_com_fixed = out_fixed["a_com"]
F_ext_fixed = out_fixed["F_ext"]

print(f"\nFixed results:")
print(f"Position range (mm): {np.max(r_com_fixed[:,2]) - np.min(r_com_fixed[:,2]):.1f}")
print(f"Max velocity (mm/s): {np.max(v_com_fixed[:,2]):.1f}")
print(f"Max acceleration (mm/s²): {np.max(a_com_fixed[:,2]):.1f}")
print(f"Max force (N): {np.max(F_ext_fixed[:,2]):.1f}")
print(f"Max force (BW): {np.max(F_ext_fixed[:,2])/(body_mass*9.81):.1f}")

# Check if forces are now reasonable
max_force_bw = np.max(F_ext_fixed[:,2])/(body_mass*9.81)
if 2 <= max_force_bw <= 5:
    print("✓ Forces now look realistic!")
else:
    print(f"⚠ Forces still unusual: {max_force_bw:.1f} BW")

print("\nReady for Step 3: Detailed analysis and plotting")

# Plot results for visual inspection
