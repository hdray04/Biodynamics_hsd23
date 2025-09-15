
import ezc3d
myjog = ezc3d.c3d("/Users/harrietdray/Biodynamics/Harriet_c3d/Jog-001/pose_filt_0.c3d")
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from src.com_force import compute_whole_body_com_fixed


#Extract 4x4 transformatino matrices for each joint across data specified 
def extract_matrices_final(myjog, labels):
    rotation_data = myjog['data']['rotations']  # Shape: (4, 4, 19, 547)
    
    print(f"Rotation data shape: {rotation_data.shape}")
    print("Structure: [4x4 matrix, 19 joints, 547 frames]")
    
    matrices_dict = {} # Dictionary to hold matrices for each joint
    n_joints = rotation_data.shape[2]  # 19 joints
    # _n_frames = rotation_data.shape[3]  # frames (unused)
    
    
    for joint_idx in range(n_joints):
        if joint_idx < len(labels):
            label = labels[joint_idx]
            joint_name = label.replace('_4X4', '')
            
            # Extract all matrices for this joint across all frames
            # Shape will be (4, 4, n_frames) -> we want (n_frames, 4, 4)
            joint_matrices = rotation_data[:, :, joint_idx, :].transpose(2, 0, 1)
            
            matrices_dict[joint_name] = joint_matrices
            
    print(f"Extracted {len(matrices_dict)} joints")
    return matrices_dict

def extract_positions_from_matrices(matrices_dict):
    positions = {}
    
    for joint_name, matrices in matrices_dict.items():
        # Extract translation component (last column, first 3 rows)
        # matrices shape: (n_frames, 4, 4)
        positions[joint_name] = matrices[:, :3, 3]  # (n_frames, 3)
    
    return positions

def _normalize_positions_keys(positions):
    """Normalize joint keys to pelvis, l_foot, r_foot when present."""
    norm = {}
    for name, arr in positions.items():
        key = name.lower()
        if 'pelvis' in key:
            norm['pelvis'] = arr
        elif ('left' in key or key.startswith('l_')) and 'foot' in key:
            norm['l_foot'] = arr
        elif ('right' in key or key.startswith('r_')) and 'foot' in key:
            norm['r_foot'] = arr
    return norm

def _infer_sampling_rate(c3d_obj, default=100.0):
    try:
        rate = c3d_obj['parameters']['POINT']['RATE']['value']
        if isinstance(rate, (list, tuple, np.ndarray)):
            return float(rate[0])
        return float(rate)
    except Exception:
        pass
    try:
        rate = c3d_obj['parameters']['ANALOG']['RATE']['value']
        if isinstance(rate, (list, tuple, np.ndarray)):
            return float(rate[0])
        return float(rate)
    except Exception:
        pass
    return float(default)

def _map_to_utils_segment_keys(positions):
    """Map Theia-style joint names to keys expected by src.utils.SEGMENTS.

    Expected keys include: pelvis, head, l_thigh, l_shank, l_foot, l_toes,
    r_thigh, r_shank, r_foot, r_toes, l_uarm, r_uarm, l_hand, r_hand.
    """
    out = {}
    def norm(s):
        return s.replace('_', '').replace('-', '').lower()
    for name, arr in positions.items():
        key = norm(name)
        if 'pelvis' in key:
            out['pelvis'] = arr
        elif 'head' in key:
            out['head'] = arr
        elif ('leftupleg' in key) or ('leftthigh' in key) or ('lthigh' in key):
            out['l_thigh'] = arr
        elif ('rightupleg' in key) or ('rightthigh' in key) or ('rthigh' in key):
            out['r_thigh'] = arr
        elif ('leftleg' in key) or ('lshank' in key) or ('leftshank' in key):
            out['l_shank'] = arr
        elif ('rightleg' in key) or ('rshank' in key) or ('rightshank' in key):
            out['r_shank'] = arr
        elif ('leftfoot' in key) or ('lfoot' in key):
            out['l_foot'] = arr
        elif ('rightfoot' in key) or ('rfoot' in key):
            out['r_foot'] = arr
        elif ('lefttoe' in key) or ('lefttoes' in key) or ('ltoes' in key):
            out['l_toes'] = arr
        elif ('righttoe' in key) or ('righttoes' in key) or ('rtoes' in key):
            out['r_toes'] = arr
        elif ('leftupperarm' in key) or ('leftarm' in key) or ('luarm' in key):
            out['l_uarm'] = arr
        elif ('rightupperarm' in key) or ('rightarm' in key) or ('ruarm' in key):
            out['r_uarm'] = arr
        elif ('lefthand' in key) or ('lhand' in key):
            out['l_hand'] = arr
        elif ('righthand' in key) or ('rhand' in key):
            out['r_hand'] = arr
    return out

def _ensure_mm(positions):
    """Ensure units are millimeters by inspecting pelvis height magnitude.
    If typical pelvis z is < 3, assume meters and convert to mm.
    """
    if 'pelvis' not in positions:
        return positions
    pel = np.asarray(positions['pelvis'])
    if pel.size == 0:
        return positions
    z_span = float(np.nanmax(pel[:, 2]) - np.nanmin(pel[:, 2]))
    if z_span < 3.0:  # likely meters
        return {k: (np.asarray(v) * 1000.0) for k, v in positions.items()}
    return positions

def _basic_hs_to_from_heights(l_foot_z, r_foot_z, fs):
    """Detect heel strikes (HS) as minima of foot Z; toe-offs (TO) via upward
    velocity zero-crossings on toe/foot Z after each HS. Returns dict with
    left/right HS/TO indices.
    """
    # Smooth
    b, a = signal.butter(3, 6.0/(fs/2.0), 'low')
    lz = signal.filtfilt(b, a, np.asarray(l_foot_z))
    rz = signal.filtfilt(b, a, np.asarray(r_foot_z))

    # HS on minima with adaptive prominence and minimum distance
    def detect_hs(z):
        p5, p95 = np.nanpercentile(z, [5, 95])
        amp = max(1e-6, float(p95 - p5))
        prominence = 0.15 * amp
        min_dist = int(max(1, round(0.30 * fs)))
        idx, _ = signal.find_peaks(-z, prominence=prominence, distance=min_dist)
        if idx.size == 0:
            idx, _ = signal.find_peaks(-z, prominence=0.5*prominence, distance=int(0.7*min_dist))
        return idx.astype(int)

    l_hs = detect_hs(lz)
    r_hs = detect_hs(rz)

    # TO as first upward vel zero-crossing after HS before next HS
    def detect_to(z, hs_idx):
        vel = np.gradient(z, 1.0/fs)
        up = np.where((vel[:-1] <= 0) & (vel[1:] > 0))[0] + 1
        to_idx = []
        for i, s in enumerate(hs_idx):
            e = hs_idx[i+1] if i+1 < len(hs_idx) else len(z)
            cand = up[(up >= s) & (up < e)]
            if cand.size:
                to_idx.append(int(cand[0]))
        return np.asarray(to_idx, int)

    l_to = detect_to(lz, l_hs)
    r_to = detect_to(rz, r_hs)

    return {
        'l_hs': l_hs, 'r_hs': r_hs,
        'l_to': l_to, 'r_to': r_to,
    }

def plot_force_time_with_events(time, Fz, l_hs=None, r_hs=None, l_to=None, r_to=None, body_mass_kg=None):
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(time, Fz, label='Predicted vertical GRF (N)')
    if body_mass_kg:
        bw = body_mass_kg * 9.81
        ax.axhline(bw, color='gray', ls='--', lw=1, label='1 BW')
    if l_hs is not None and len(l_hs):
        ax.scatter(time[l_hs], Fz[l_hs], marker='v', c='C1', label='L HS')
    if r_hs is not None and len(r_hs):
        ax.scatter(time[r_hs], Fz[r_hs], marker='v', c='C2', label='R HS')
    if l_to is not None and len(l_to):
        ax.scatter(time[l_to], Fz[l_to], marker='^', c='C1', label='L TO')
    if r_to is not None and len(r_to):
        ax.scatter(time[r_to], Fz[r_to], marker='^', c='C2', label='R TO')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Force (N)')
    ax.set_title('Force–time curve from COM (jogging)')
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend()
    plt.tight_layout()
    return fig, ax

def extract_angles(myjog, labels2):
    angle_data = myjog['data']['points']  # Shape: (3, N_labels, frames)
    angle_indices = {
    'left_knee': labels2.index('LeftKneeAngles_Theia'),
    'right_knee': labels2.index('RightKneeAngles_Theia'),
    'left_hip': labels2.index('LeftHipAngles_Theia'),
    'right_hip': labels2.index('RightHipAngles_Theia'),
    'left_fp': labels2.index('LeftFootProgressionAngles_Theia'),
    'right_fp': labels2.index('RightFootProgressionAngles_Theia')
    }

    angles = {key: {'Sagittal': [], 'Frontal': [], 'Transverse': []} for key in angle_indices.keys()}
    
    n_frames = angle_data.shape[2]
    
    for key, idx in angle_indices.items():
        # Extract angles for all three planes
        angles[key]['Sagittal'] = [angle_data[0, idx, :n_frames]]      # X-rotation
        angles[key]['Frontal'] = [angle_data[1, idx, :n_frames]]       # Y-rotation  
        angles[key]['Transverse'] = [angle_data[2, idx, :n_frames]]    # Z-rotation
    
    return angles


def analyse_walk_movement(positions, angles):
    required_joints = ['pelvis', 'l_foot', 'r_foot']
    missing = [joint for joint in required_joints if joint not in positions]
    if missing:
        print(f"Missing joints for analysis: {missing}")
        print(f"Available joints: {list(positions.keys())}")
        return None
        
    pelvis = positions['pelvis']
    l_foot = positions['l_foot']
    r_foot = positions['r_foot']
    
    frames = len(pelvis)
    print(f"Analyzing {frames} frames of gait data")
    print("="*60)

    # KNEE ANALYSIS
    l_knee_sag = angles['left_knee']['Sagittal']
    r_knee_sag = angles['right_knee']['Sagittal']
    l_knee_front = angles['left_knee']['Frontal']
    r_knee_front = angles['right_knee']['Frontal']

    print("KNEE ANGLES (Sagittal Plane - Flexion/Extension):")
    print(f"   Left  - Max: {np.max(l_knee_sag):6.1f}° | Min: {np.min(l_knee_sag):6.1f}° | ROM: {np.max(l_knee_sag) - np.min(l_knee_sag):5.1f}°")
    print(f"   Right - Max: {np.max(r_knee_sag):6.1f}° | Min: {np.min(r_knee_sag):6.1f}° | ROM: {np.max(r_knee_sag) - np.min(r_knee_sag):5.1f}°")
    
    print("KNEE ANGLES (Frontal Plane - Varus/Valgus):")
    print(f"   Left ROM:  {np.max(l_knee_front) - np.min(l_knee_front):5.1f}°")
    print(f"   Right ROM: {np.max(r_knee_front) - np.min(r_knee_front):5.1f}°")

    # HIP ANALYSIS
    l_hip_sag = angles['left_hip']['Sagittal']
    r_hip_sag = angles['right_hip']['Sagittal']
    l_hip_front = angles['left_hip']['Frontal']
    r_hip_front = angles['right_hip']['Frontal']
    
    print("\nHIP ANGLES (Sagittal Plane - Flexion/Extension):")
    print(f"   Left  - Max: {np.max(l_hip_sag):6.1f}° | Min: {np.min(l_hip_sag):6.1f}° | ROM: {np.max(l_hip_sag) - np.min(l_hip_sag):5.1f}°")
    print(f"   Right - Max: {np.max(r_hip_sag):6.1f}° | Min: {np.min(r_hip_sag):6.1f}° | ROM: {np.max(r_hip_sag) - np.min(r_hip_sag):5.1f}°")
    
    print("HIP ANGLES (Frontal Plane - Abduction/Adduction):")
    print(f"   Left ROM:  {np.max(l_hip_front) - np.min(l_hip_front):5.1f}°")
    print(f"   Right ROM: {np.max(r_hip_front) - np.min(r_hip_front):5.1f}°")
    
    # FOOT PROGRESSION
    l_fp = angles['left_fp']['Transverse']
    r_fp = angles['right_fp']['Transverse']
    
    print("\nFOOT PROGRESSION (Transverse Plane - Internal/External Rotation):")
    print(f"   Left  - Mean: {np.mean(l_fp):6.1f}° | Std: {np.std(l_fp):5.1f}°")
    print(f"   Right - Mean: {np.mean(r_fp):6.1f}° | Std: {np.std(r_fp):5.1f}°")
    
    # SYMMETRY ANALYSIS
    print("\nSYMMETRY ANALYSIS:")
    knee_rom_l = np.max(l_knee_sag) - np.min(l_knee_sag)
    knee_rom_r = np.max(r_knee_sag) - np.min(r_knee_sag)
    knee_symmetry = min(knee_rom_l, knee_rom_r) / max(knee_rom_l, knee_rom_r)
    
    hip_rom_l = np.max(l_hip_sag) - np.min(l_hip_sag)
    hip_rom_r = np.max(r_hip_sag) - np.min(r_hip_sag)
    hip_symmetry = min(hip_rom_l, hip_rom_r) / max(hip_rom_l, hip_rom_r)
    
    print(f"   Knee ROM Symmetry: {knee_symmetry:.3f} (1.0 = perfect)")
    print(f"   Hip ROM Symmetry:  {hip_symmetry:.3f} (1.0 = perfect)")
    
    # CLINICAL ASSESSMENT
    print("\nCLINICAL ASSESSMENT:")
    warnings = []
    
    if np.min(l_knee_sag) < -5 or np.min(r_knee_sag) < -5:
        warnings.append("Knee hyperextension detected")
    if np.max(l_knee_sag) > 70 or np.max(r_knee_sag) > 70:
        warnings.append("Excessive knee flexion detected")
    if knee_rom_l < 30 or knee_rom_r < 30:
        warnings.append("Limited knee ROM detected")
    if abs(np.mean(l_fp)) > 15 or abs(np.mean(r_fp)) > 15:
        warnings.append("Abnormal foot progression detected")
    if knee_symmetry < 0.8 or hip_symmetry < 0.8:
        warnings.append("Significant asymmetry detected")
    
    if warnings:
        for warning in warnings:
            print(f"   WARNING: {warning}")
    else:
        print("   No major gait deviations detected")
    
    # BASIC SPATIAL METRICS
    print("\nSPATIAL METRICS:")
    pelvis_forward = pelvis[:, 1]  # Y-axis (forward direction)
    total_distance = abs(pelvis_forward[-1] - pelvis_forward[0])
    avg_speed = total_distance / frames
    
    print(f"   Total Distance: {total_distance:7.1f} mm ({total_distance/1000:.2f} m)")
    print(f"   Average Speed:  {avg_speed:7.2f} mm/frame")
    
    # Step width
    step_widths = []
    for frame in range(frames):
        width = abs(l_foot[frame, 0] - r_foot[frame, 0])
        step_widths.append(width)
    
    avg_step_width = np.mean(step_widths)
    print(f"   Average Step Width: {avg_step_width:6.1f} mm")
    
    print("="*60)
    
    # RETURN RESULTS
    return {
        'knee_symmetry': knee_symmetry,
        'hip_symmetry': hip_symmetry,
        'total_distance': total_distance,
        'avg_speed': avg_speed,
        'avg_step_width': avg_step_width,
        'warnings': warnings
    }

"""
Execution: analysis + COM-based force and gait cycle
"""
# Labels
labels_rot = myjog['parameters']['ROTATION']['LABELS']['value']
labels_pts = myjog['parameters']['POINT']['LABELS']['value']
fs = _infer_sampling_rate(myjog, default=100.0)
BODY_MASS_KG = 54.0

# Positions from ROTATION 4x4 matrices
matrices = extract_matrices_final(myjog, labels_rot)
positions_all = extract_positions_from_matrices(matrices)
positions = _normalize_positions_keys(positions_all)

# Angles (if available in labels)
try:
    angles = extract_angles(myjog, labels_pts)
except Exception as e:
    print("Angle extraction issue:", e)
    angles = None

# COM-derived force
try:
    seg_positions = _map_to_utils_segment_keys(positions_all)
    seg_positions = _ensure_mm(seg_positions)
    com_out = compute_whole_body_com_fixed(seg_positions, BODY_MASS_KG, fs, cutoff_freq=6.0)
    F_ext_smooth = com_out['F_ext_smooth']
    # vertical component; Theia Z is usually vertical
    Fz = F_ext_smooth[:, 2]
    time = np.arange(Fz.shape[0]) / fs
except Exception as e:
    print("COM/force computation failed:", e)
    Fz = None

# Gait events from foot heights
if {'l_foot','r_foot'}.issubset(positions):
    lz = positions['l_foot'][:, 2]
    rz = positions['r_foot'][:, 2]
    ev = _basic_hs_to_from_heights(lz, rz, fs)
    l_hs, r_hs, l_to, r_to = ev['l_hs'], ev['r_hs'], ev['l_to'], ev['r_to']
else:
    l_hs = r_hs = l_to = r_to = np.array([], int)

# Print basic summary
print("Processing gait data...")
if angles is not None:
    res = analyse_walk_movement(positions, angles)
    if res:
        print("\nSUMMARY:")
        print(f"Knee symmetry: {res['knee_symmetry']:.3f}")
        print(f"Hip symmetry: {res['hip_symmetry']:.3f}")
        print(f"Jogging speed (pelvis): {res['avg_speed']:.2f} mm/frame")

# Plot force–time with events
if Fz is not None:
    plot_force_time_with_events(time, Fz, l_hs, r_hs, l_to, r_to, body_mass_kg=BODY_MASS_KG)
    plt.show()

def analyse_walk_movement(positions, angles):
    required_joints = ['pelvis', 'l_foot', 'r_foot']
    missing = [joint for joint in required_joints if joint not in positions]
    if missing:
        print(f"Missing joints for analysis: {missing}")
        print(f"Available joints: {list(positions.keys())}")
        return None
    pelvis = positions['pelvis']
    l_foot = positions['l_foot']
    r_foot = positions['r_foot']
    
    frames = len(pelvis)
    print(f"Analyzing {frames} frames of gait data")

    l_knee_sag = angles['left_knee']['Sagittal']
    r_knee_sag = angles['right_knee']['Sagittal']
    l_knee_front = angles['left_knee']['Frontal']
    r_knee_front = angles['right_knee']['Frontal']

    print("KNEE ANGLES (Sagittal Plane - Flexion/Extension):")
    print(f"   Left  - Max: {np.max(l_knee_sag):6.1f}° | Min: {np.min(l_knee_sag):6.1f}° | ROM: {np.max(l_knee_sag) - np.min(l_knee_sag):5.1f}°")
    print(f"   Right - Max: {np.max(r_knee_sag):6.1f}° | Min: {np.min(r_knee_sag):6.1f}° | ROM: {np.max(r_knee_sag) - np.min(r_knee_sag):5.1f}°")
    
    print("KNEE ANGLES (Frontal Plane - Varus/Valgus):")
    print(f"   Left ROM:  {np.max(l_knee_front) - np.min(l_knee_front):5.1f}°")
    print(f"   Right ROM: {np.max(r_knee_front) - np.min(r_knee_front):5.1f}°")

    # HIP ANALYSIS
    l_hip_sag = angles['left_hip']['Sagittal']
    r_hip_sag = angles['right_hip']['Sagittal']
    l_hip_front = angles['left_hip']['Frontal']
    r_hip_front = angles['right_hip']['Frontal']
    
    print("\nHIP ANGLES (Sagittal Plane - Flexion/Extension):")
    print(f"   Left  - Max: {np.max(l_hip_sag):6.1f}° | Min: {np.min(l_hip_sag):6.1f}° | ROM: {np.max(l_hip_sag) - np.min(l_hip_sag):5.1f}°")
    print(f"   Right - Max: {np.max(r_hip_sag):6.1f}° | Min: {np.min(r_hip_sag):6.1f}° | ROM: {np.max(r_hip_sag) - np.min(r_hip_sag):5.1f}°")
    
    print("HIP ANGLES (Frontal Plane - Abduction/Adduction):")
    print(f"   Left ROM:  {np.max(l_hip_front) - np.min(l_hip_front):5.1f}°")
    print(f"   Right ROM: {np.max(r_hip_front) - np.min(r_hip_front):5.1f}°")
    
    # FOOT PROGRESSION
    l_fp = angles['left_fp']['Transverse']
    r_fp = angles['right_fp']['Transverse']
    
    print("\nFOOT PROGRESSION (Transverse Plane - Internal/External Rotation):")
    print(f"   Left  - Mean: {np.mean(l_fp):6.1f}° | Std: {np.std(l_fp):5.1f}°")
    print(f"   Right - Mean: {np.mean(r_fp):6.1f}° | Std: {np.std(r_fp):5.1f}°")
    
    # SYMMETRY ANALYSIS
    print("\nSYMMETRY ANALYSIS:")
    knee_rom_l = np.max(l_knee_sag) - np.min(l_knee_sag)
    knee_rom_r = np.max(r_knee_sag) - np.min(r_knee_sag)
    knee_symmetry = min(knee_rom_l, knee_rom_r) / max(knee_rom_l, knee_rom_r)
    
    hip_rom_l = np.max(l_hip_sag) - np.min(l_hip_sag)
    hip_rom_r = np.max(r_hip_sag) - np.min(r_hip_sag)
    hip_symmetry = min(hip_rom_l, hip_rom_r) / max(hip_rom_l, hip_rom_r)
    
    print(f"   Knee ROM Symmetry: {knee_symmetry:.3f} (1.0 = perfect)")
    print(f"   Hip ROM Symmetry:  {hip_symmetry:.3f} (1.0 = perfect)")
