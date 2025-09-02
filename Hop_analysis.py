import ezc3d
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# COM utilities
from src.com_force import compute_whole_body_com_fixed


hop_single = ezc3d.c3d("/Users/harrietdray/Biodynamics/Harriet_c3d/HopSingle-001/pose_filt_0.c3d")
triple_hop_1 = ezc3d.c3d("/Users/harrietdray/Biodynamics/Harriet_c3d/HopTriple-001/pose_filt_0.c3d")
triple_hop_2 = ezc3d.c3d("/Users/harrietdray/Biodynamics/Harriet_c3d/HopTriple-002/pose_filt_0.c3d")

angle_data_trials = [hop_single['data']['points'], triple_hop_1['data']['points'], triple_hop_2['data']['points']]
rotation_data_trials = [hop_single['data']['rotations'], triple_hop_1['data']['rotations'], triple_hop_2['data']['rotations']]
hop_data = [hop_single, triple_hop_1, triple_hop_2]
labels = hop_data[0]['parameters']['POINT']['LABELS']['value'] # List of angle names
labels_rotation = hop_data[0]['parameters']['ROTATION']['LABELS']['value']
units = hop_data[0]['parameters']['POINT']['UNITS']['value']
print("Units used:", units)

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

    out = {}
    for trial_idx, angle_data in enumerate(angle_data_trials):
        n_frames = angle_data.shape[2]
        one = {}
        for joint_name, joint_idx in angle_indices.items():
            sagittal = angle_data[0, joint_idx, :n_frames]
            frontal = angle_data[1, joint_idx, :n_frames]
            transverse = angle_data[2, joint_idx, :n_frames]
            one[joint_name] = {
                'sagittal': sagittal,
                'frontal': frontal,
                'transverse': transverse,
            }
        out[f'trial_{trial_idx + 1}'] = one
    return out


# Extract 3D angles for all trials
all_angles_3d = extract_3d_angles(angle_data_trials, labels)



def extract_matrices(rotation_data_trials, labels_rotation):# Shape: (4, 4, 19, 547)
    all_matrices = {} # Dictionary to hold matrices for each joint
    
    for trial_idx, rotation_data_ in enumerate(rotation_data_trials):
        rotation_data = hop_data[trial_idx]['data']['rotations']  # Extract rotation data for the current trial
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
    
all_matrices = extract_matrices(rotation_data_trials, labels_rotation)

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


all_positions = extract_positions_from_matrices(all_matrices)

fs = 100  # Hz

def butter_low(x, fs, fc=8, order=2):
    b,a = butter(order, fc/(fs/2), btype='low')
    return filtfilt(b,a,x)

def detect_to_ic(foot_z, fs, thresh=None):
    """Return first take-off then landing frame indices from vertical foot pos.

    Finds the earliest take-off with a subsequent initial contact.
    """
    z = butter_low(foot_z, fs, fc=8)
    if thresh is None:
        thresh = np.percentile(z, 20) + 0.005
    stance = z < thresh
    ds = np.diff(stance.astype(int), prepend=stance[0])
    to_idx = np.where(ds == -1)[0]  # foot leaves ground
    ic_idx = np.where(ds == 1)[0]   # foot contacts ground
    if len(to_idx) == 0 or len(ic_idx) == 0:
        return None, None
    # Pair the first TO with the first IC that occurs after it
    for t in to_idx:
        later_ic = ic_idx[ic_idx > t]
        if later_ic.size:
            return int(t), int(later_ic[0])
    return None, None

def hop_time_and_distance(single_positions):
    """
    single_positions: dict from extract_positions_from_matrices()['trial_1']
                      e.g. single_positions['RightFoot'] -> (N,3)
    """
    # Prefer right foot if available, else left
    foot_key = 'r_foot' if 'r_foot' in single_positions else ('RightFoot' if 'RightFoot' in single_positions else None)
    if foot_key is None:
        foot_key = 'l_foot' if 'l_foot' in single_positions else ('LeftFoot' if 'LeftFoot' in single_positions else None)
    if foot_key is None:
        return None
    foot = single_positions[foot_key]
    foot_z = foot[:,2]
    to_i, ic_i = detect_to_ic(foot_z, fs)
    if to_i is None: 
        return None
    
    flight_time = (ic_i - to_i)/fs
    distance_mm = foot[ic_i,0] - foot[to_i,0]  # assumes X is forward axis, in mm
    distance = float(distance_mm) / 1000.0

    return {'TO_frame': to_i,
            'IC_frame': ic_i,
            'flight_time_s': flight_time,
            'distance_m': distance}

def detect_all_hops(single_positions, fs, min_flight_s=0.12, max_flight_s=1.5, min_stance_s=0.10,
                    low_mm=10.0, high_mm=25.0):
    """Detect all hops using hysteresis on foot height relative to ground.

    - Positions may be in m or mm; we internally convert to mm.
    - low_mm/high_mm define hysteresis thresholds on foot height above baseline.
    - Debounces short aerial/stance segments.
    Returns list of (to_idx, ic_idx).
    """
    positions_mm = infer_positions_unit_scale_mm(single_positions)
    # Choose foot
    foot_key = 'r_foot' if 'r_foot' in positions_mm else ('RightFoot' if 'RightFoot' in positions_mm else None)
    if foot_key is None:
        foot_key = 'l_foot' if 'l_foot' in positions_mm else ('LeftFoot' if 'LeftFoot' in positions_mm else None)
    if foot_key is None:
        return []
    z = butter_low(positions_mm[foot_key][:, 2], fs, fc=8)
    baseline = np.percentile(z, 5)
    rel = z - baseline
    n = rel.shape[0]

    # State machine with hysteresis
    in_air = False
    last_state_change = 0
    tos = []
    ics = []
    for i in range(n):
        if not in_air:
            # on ground; look for take-off when exceeding high threshold
            if rel[i] > high_mm:
                # ensure prior stance duration
                if (i - last_state_change) / fs >= min_stance_s:
                    in_air = True
                    tos.append(i)
                    last_state_change = i
        else:
            # in air; look for landing when dropping below low threshold
            if rel[i] < low_mm:
                # ensure flight duration
                if (i - last_state_change) / fs >= min_flight_s:
                    in_air = False
                    ics.append(i)
                    last_state_change = i
    # Pair TO->IC
    pairs = []
    j = 0
    for t in tos:
        while j < len(ics) and ics[j] <= t:
            j += 1
        if j < len(ics):
            ic = ics[j]
            dur = (ic - t) / fs
            if min_flight_s <= dur <= max_flight_s:
                pairs.append((t, ic))
            j += 1
    return pairs

def infer_positions_unit_scale_mm(single_positions):
    """Return a copy of positions dict scaled to mm, inferring unit from magnitude.

    Heuristic: if typical pelvis/COM height is < 3, assume meters -> convert to mm.
    """
    keys_priority = ['pelvis', 'Pelvis', 'r_foot', 'RightFoot', 'l_foot', 'LeftFoot']
    probe = None
    for k in keys_priority:
        if k in single_positions:
            probe = single_positions[k]
            break
    if probe is None:
        # Fallback to any entry
        probe = next(iter(single_positions.values()))
    span = float(np.nanmax(probe[:, 2]) - np.nanmin(probe[:, 2]))
    scale = 1000.0 if span < 3.0 else 1.0  # <3 implies meters
    if scale == 1.0:
        return single_positions
    scaled = {k: (v * scale) for k, v in single_positions.items()}
    return scaled

def hop_metrics_from_com(single_positions, fs, body_mass_kg=54.0):
    """Compute hop metrics using COM.

    Expects single_positions to include keys matching src.utils.SEGMENTS joints
    (e.g., 'pelvis', 'head', 'r_thigh', 'r_shank', 'r_foot', 'r_toes', etc.).

    Returns a dict with event frames, flight time, COM jump height, peak stats.
    """
    # 1) Detect TO/IC using foot kinematics for robustness
    events = hop_time_and_distance(single_positions)
    if not events:
        return None

    to_i = events['TO_frame']
    ic_i = events['IC_frame']
    # Guard against invalid or empty flight window
    if to_i is None or ic_i is None:
        return None
    n_frames = next(iter(single_positions.values())).shape[0]
    if not (0 <= to_i < ic_i < n_frames):
        return None

    # 2) Compute COM trajectory (ensure positions are in mm for src.com_force)
    positions_mm = infer_positions_unit_scale_mm(single_positions)
    out = compute_whole_body_com_fixed(positions_mm, body_mass_kg, fs, cutoff_freq=6.0)
    r_com = out['r_com']  # shape (N, 3), mm

    # 3) Jump height from COM during flight
    com_z = r_com[:, 2]
    com_z_flight = com_z[to_i:ic_i+1]
    if com_z_flight.size == 0:
        return None
    peak_idx_rel = int(np.argmax(com_z_flight))
    peak_idx = to_i + peak_idx_rel
    com_height_jump_mm = float(com_z[peak_idx] - com_z[to_i])
    com_height_jump_m = com_height_jump_mm / 1000.0

    # 4) Additional metrics
    t_to = to_i / fs
    t_ic = ic_i / fs
    t_peak = peak_idx / fs

    # Horizontal displacement: use foot progression (X-axis) rather than COM
    horiz_disp_m = events['distance_m']

    v_com = out['v_com']  # mm/s
    a_com = out['a_com']  # mm/s^2
    peak_vz_mm_s = float(np.max(v_com[to_i:ic_i+1, 2]))
    min_vz_mm_s = float(np.min(v_com[to_i:ic_i+1, 2]))
    peak_az_mm_s2 = float(np.max(a_com[to_i:ic_i+1, 2]))
    min_az_mm_s2 = float(np.min(a_com[to_i:ic_i+1, 2]))

    return {
        'TO_frame': to_i,
        'IC_frame': ic_i,
        'TO_time_s': t_to,
        'IC_time_s': t_ic,
        'flight_time_s': events['flight_time_s'],
        'com_jump_height_mm': com_height_jump_mm,
        'com_jump_height_m': com_height_jump_m,
        'com_peak_frame': peak_idx,
        'com_peak_time_s': t_peak,
        'com_takeoff_height_mm': float(com_z[to_i]),
        'com_landing_height_mm': float(com_z[ic_i]),
        'foot_horizontal_displacement_m': horiz_disp_m,
        'peak_vz_mm_s': peak_vz_mm_s,
        'min_vz_mm_s': min_vz_mm_s,
        'peak_az_mm_s2': peak_az_mm_s2,
        'min_az_mm_s2': min_az_mm_s2,
    }

# Multi-hop metrics per trial
def hop_metrics_per_hop(single_positions, fs, body_mass_kg=54.0):
    positions_mm = infer_positions_unit_scale_mm(single_positions)
    out = compute_whole_body_com_fixed(positions_mm, body_mass_kg, fs, cutoff_freq=6.0)
    r_com = out['r_com']
    v_com = out['v_com']
    a_com = out['a_com']

    pairs = detect_all_hops(single_positions, fs)
    # Choose foot for horizontal displacement (X-axis)
    foot_key = 'r_foot' if 'r_foot' in single_positions else ('RightFoot' if 'RightFoot' in single_positions else None)
    if foot_key is None:
        foot_key = 'l_foot' if 'l_foot' in single_positions else ('LeftFoot' if 'LeftFoot' in single_positions else None)
    foot_pos = single_positions.get(foot_key, None)
    results = []
    for to_i, ic_i in pairs:
        com_z = r_com[:, 2]
        if ic_i <= to_i or to_i < 0 or ic_i > com_z.shape[0] - 1:
            continue
        flight = slice(to_i, ic_i + 1)
        if (ic_i - to_i) <= 0:
            continue
        com_z_flight = com_z[flight]
        if com_z_flight.size == 0:
            continue
        peak_idx_rel = int(np.argmax(com_z_flight))
        peak_idx = to_i + peak_idx_rel
        com_height_jump_m = float(com_z[peak_idx] - com_z[to_i]) / 1000.0
        # Foot-based horizontal displacement along forward X-axis (meters)
        if foot_pos is not None:
            horiz_disp_m = float(foot_pos[ic_i, 0] - foot_pos[to_i, 0]) / 1000.0
        else:
            horiz_disp_m = float(r_com[ic_i, 0] - r_com[to_i, 0]) / 1000.0

        results.append({
            'TO_frame': int(to_i),
            'IC_frame': int(ic_i),
            'TO_time_s': to_i / fs,
            'IC_time_s': ic_i / fs,
            'flight_time_s': (ic_i - to_i) / fs,
            'com_jump_height_m': com_height_jump_m,
            'com_peak_frame': int(peak_idx),
            'com_peak_time_s': peak_idx / fs,
            'com_takeoff_height_mm': float(com_z[to_i]),
            'com_landing_height_mm': float(com_z[ic_i]),
            'foot_horizontal_displacement_m': horiz_disp_m,
            'peak_vz_mm_s': float(np.max(v_com[flight, 2])),
            'min_vz_mm_s': float(np.min(v_com[flight, 2])),
            'peak_az_mm_s2': float(np.max(a_com[flight, 2])),
            'min_az_mm_s2': float(np.min(a_com[flight, 2])),
        })
    return results

# Example: compute metrics for each trial
per_trial_hops = {}
name_map = {
    'trial_1': 'single_hop',
    'trial_2': 'triple_hop_1',
    'trial_3': 'triple_hop_2',
}
expected_counts = {
    'single_hop': 1,
    'triple_hop_1': 3,
    'triple_hop_2': 3,
}
for tname, pos in all_positions.items():
    label = name_map.get(tname, tname)
    hops = hop_metrics_per_hop(pos, fs=fs, body_mass_kg=54.0)
    # If too many hops detected, keep the first N expected
    exp = expected_counts.get(label)
    if exp is not None and len(hops) > exp:
        hops = hops[:exp]
    per_trial_hops[label] = hops

# Quick printout per hop
for t, hops in per_trial_hops.items():
    print(f"\n{t}:")
    if not hops:
        print("  No valid hops detected.")
        continue
    for i, m in enumerate(hops, 1):
        print(f"  Hop {i}:")
        print(f"    Flight time: {m['flight_time_s']:.3f} s")
        print(f"    COM jump height: {m['com_jump_height_m']:.3f} m")
        print(f"    Horizontal displacement (COM): {m['com_horizontal_displacement_m']:.3f} m")
        print(f"    Peak COM at {m['com_peak_time_s']:.3f} s (frame {m['com_peak_frame']})")
