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
# --- COM calculation ---
def compute_whole_body_com(joints, body_mass, fs, g_vec=np.array([0, 0, -9.81])):
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

    nyquist = fs / 2
    normalized_cutoff = cutoff_freq / nyquist
    
    # Design Butterworth filter (4th order is standard)
    b, a = signal.butter(4, normalized_cutoff, 'low')
    
    # Apply zero-phase filtering (filtfilt) to avoid phase lag
    r_com_filtered = signal.filtfilt(b, a, r_com_raw, axis=0)
    
    print(f"Applied {cutoff_freq}Hz low-pass filter")

    dt = 1.0 / fs
    v_com = np.gradient(r_com_raw, dt, axis=0)
    a_com = np.gradient(v_com, dt, axis=0)
    g_vec = np.asarray(g_vec)
    if g_vec.ndim == 0:  # If it's a scalar
        g_vec = np.array([0, 0, g_vec])  # Assume it's the z-component
    F_ext = total_mass * (a_com - g_vec[None, :])

    v_com = np.zeros_like(r_com_filtered)
    v_com[1:-1] = (r_com_filtered[2:] - r_com_filtered[:-2]) / (2 * dt)
    v_com[0] = (r_com_filtered[1] - r_com_filtered[0]) / dt  # Forward difference for first point
    v_com[-1] = (r_com_filtered[-1] - r_com_filtered[-2]) / dt  # Backward difference for last point
    
    # Acceleration using central differences on velocity
    a_com = np.zeros_like(v_com)
    a_com[1:-1] = (v_com[2:] - v_com[:-2]) / (2 * dt)
    a_com[0] = (v_com[1] - v_com[0]) / dt
    a_com[-1] = (v_com[-1] - v_com[-2]) / dt

    
    F_ext_smooth = signal.filtfilt(b, a, F_ext, axis=0)
    
    return {
        "r_com_raw": r_com_raw,
        "r_com": r_com_filtered,
        "v_com": v_com,
        "a_com": a_com,
        "F_ext": F_ext,
        "F_ext_smooth": F_ext_smooth,
        "filter_info": {"cutoff": cutoff_freq, "fs": fs}
    }

# --- Parameters ---
fs = float(cmj_1['parameters']['POINT']['RATE']['value'][0])  # Hz
body_mass = 54.0  # kg
g_vec = np.array([0, 0, -9.81])

# --- Run ---
out = compute_whole_body_com(positions, body_mass, fs, g_vec=g_vec)

r_com = out["r_com"]
v_com = out["v_com"]
a_com = out["a_com"]
F_ext = out["F_ext"]

# Apply low-pass filter (typically 6-12 Hz cutoff for human movement

# Convert from mm to meters for standard biomechanics units
r_com_m = r_com / 1000.0  # mm to m
v_com_ms = v_com / 1000.0  # mm/s to m/s  
a_com_ms2 = a_com / 1000.0  # mm/s² to m/s²
# F_ext is already in Newtons (calculated from mass in kg and acceleration)


"""
def verify_standing_com(positions, r_com, body_mass=54.0, actual_height_cm=162):
 
    print("=== STANDING COM VERIFICATION ===\n")
    
    # 1. Find quiet standing period (low COM movement)
    r_com_mm = r_com  # Keep in mm for this analysis
    
    # Calculate COM movement magnitude
    com_movement = np.sqrt(np.sum(np.diff(r_com_mm, axis=0)**2, axis=1))
    
    # Find the quietest period (lowest movement)
    window_size = int(1.0 * 100)  # 1 second window at 100Hz
    movement_smoothed = np.convolve(com_movement, np.ones(window_size)/window_size, mode='valid')
    quiet_start = np.argmin(movement_smoothed)
    quiet_end = quiet_start + window_size
    
    print(f"Identified quiet standing period: frames {quiet_start} to {quiet_end}")
    
    # 2. Calculate mean COM during quiet standing
    quiet_com = np.mean(r_com_mm[quiet_start:quiet_end], axis=0)
    quiet_std = np.std(r_com_mm[quiet_start:quiet_end], axis=0)
    
    print(f"Standing COM position (mm):")
    print(f"  X (ML): {quiet_com[0]:.1f} ± {quiet_std[0]:.1f}")
    print(f"  Y (AP): {quiet_com[1]:.1f} ± {quiet_std[1]:.1f}") 
    print(f"  Z (Vertical): {quiet_com[2]:.1f} ± {quiet_std[2]:.1f}")
    print()
    
    # 3. Check joint positions during quiet standing
    print("Key joint positions during quiet standing (mm):")
    key_joints = ['pelvis', 'head', 'l_foot', 'r_foot']
    joint_heights = {}
    
    for joint in key_joints:
        if joint in positions:
            joint_pos = np.mean(positions[joint][quiet_start:quiet_end], axis=0)
            joint_heights[joint] = joint_pos[2]  # Z-coordinate
            print(f"  {joint}: Z = {joint_pos[2]:.1f} mm")
    
    # 4. Estimate body height from joint positions
    if 'head' in joint_heights and 'l_foot' in joint_heights:
        estimated_height_mm = joint_heights['head'] - min(joint_heights['l_foot'], 
                                                          joint_heights.get('r_foot', joint_heights['l_foot']))
        estimated_height_m = estimated_height_mm / 1000.0
        print(f"\nEstimated body height: {estimated_height_m:.2f} m ({estimated_height_mm:.0f} mm)")
        
        # 5. COM height as percentage of body height
        foot_height = min([h for joint, h in joint_heights.items() if 'foot' in joint])
        com_height_above_ground = quiet_com[2] - foot_height
        com_height_percentage = com_height_above_ground / estimated_height_mm * 100
        
        print(f"COM height above ground: {com_height_above_ground:.1f} mm ({com_height_above_ground/1000:.3f} m)")
        print(f"COM height as % of body height: {com_height_percentage:.1f}%")
        print("  Expected range: 55-60% for adults")
        
        # Validation
        if 50 <= com_height_percentage <= 65:
            print("  ✓ COM height percentage looks reasonable")
        else:
            print("  ⚠ COM height percentage seems unusual")
            
    # 6. Check COM stability (sway)
    sway_range = np.max(r_com_mm[quiet_start:quiet_end], axis=0) - np.min(r_com_mm[quiet_start:quiet_end], axis=0)
    print(f"\nPostural sway during quiet standing (mm):")
    print(f"  ML sway range: {sway_range[0]:.1f} mm")
    print(f"  AP sway range: {sway_range[1]:.1f} mm") 
    print(f"  Vertical sway range: {sway_range[2]:.1f} mm")
    print("  Typical quiet standing sway: <10-20mm in each direction")
    
    if sway_range[0] < 30 and sway_range[1] < 30:
        print("  ✓ Postural sway looks reasonable")
    else:
        print("  ⚠ Postural sway seems excessive - check data quality")
    
    # 7. Visual verification
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Time series of COM position
    time = np.arange(len(r_com_mm)) / 100.0
    
    axes[0,0].plot(time, r_com_mm[:, 0], 'r-', alpha=0.7)
    axes[0,0].axvspan(quiet_start/100, quiet_end/100, alpha=0.3, color='gray', label='Quiet standing')
    axes[0,0].set_ylabel('ML Position (mm)')
    axes[0,0].set_title('Mediolateral COM')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    axes[0,1].plot(time, r_com_mm[:, 1], 'g-', alpha=0.7)
    axes[0,1].axvspan(quiet_start/100, quiet_end/100, alpha=0.3, color='gray')
    axes[0,1].set_ylabel('AP Position (mm)')
    axes[0,1].set_title('Anteroposterior COM')
    axes[0,1].grid(True, alpha=0.3)
    
    axes[1,0].plot(time, r_com_mm[:, 2], 'b-', alpha=0.7)
    axes[1,0].axvspan(quiet_start/100, quiet_end/100, alpha=0.3, color='gray')
    axes[1,0].set_ylabel('Vertical Position (mm)')
    axes[1,0].set_xlabel('Time (s)')
    axes[1,0].set_title('Vertical COM')
    axes[1,0].grid(True, alpha=0.3)
    
    # COM movement magnitude
    axes[1,1].plot(time[1:], com_movement, 'k-', alpha=0.7)
    axes[1,1].axvspan(quiet_start/100, quiet_end/100, alpha=0.3, color='gray')
    axes[1,1].set_ylabel('COM Movement (mm)')
    axes[1,1].set_xlabel('Time (s)')
    axes[1,1].set_title('COM Movement Magnitude')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.suptitle('Standing COM Verification', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    return {
        'quiet_period': (quiet_start, quiet_end),
        'standing_com': quiet_com,
        'com_height_percentage': com_height_percentage if 'head' in joint_heights else None,
        'sway_range': sway_range
    }
verification_results = verify_standing_com(positions, r_com, body_mass=54.0, actual_height_cm=162)
"""

def analyze_jump_performance(results, body_mass, fs):
    """
    Extract key jump performance metrics
    """
    r_com = results["r_com"] / 1000.0  # Convert to meters
    v_com = results["v_com"] / 1000.0  # Convert to m/s
    a_com = results["a_com"] / 1000.0  # Convert to m/s²
    F_ext = results["F_ext_smooth"]  # Already in Newtons
    
    time = np.arange(len(r_com)) / fs
    
    # Find key phases
    vertical_vel = v_com[:, 2]
    vertical_pos = r_com[:, 2]
    vertical_force = F_ext[:, 2]
    
    # 1. Find takeoff and landing
    # Takeoff: first time vertical velocity becomes positive and stays positive
    positive_vel_idx = np.where(vertical_vel > 0.1)[0]  # 0.1 m/s threshold
    takeoff_idx = positive_vel_idx[0] if len(positive_vel_idx) > 0 else None
    
    # Landing: first time after takeoff that velocity becomes negative
    if takeoff_idx is not None:
        post_takeoff = np.where((time > time[takeoff_idx] + 0.1) & (vertical_vel < -0.1))[0]
        landing_idx = post_takeoff[0] if len(post_takeoff) > 0 else None
    else:
        landing_idx = None
    
    # 2. Calculate performance metrics
    jump_height = np.max(vertical_pos) - np.min(vertical_pos)
    peak_velocity = np.max(vertical_vel)
    peak_force = np.max(vertical_force)
    peak_force_bw = peak_force / (body_mass * 9.81)
    
    # Flight time
    if takeoff_idx is not None and landing_idx is not None:
        flight_time = time[landing_idx] - time[takeoff_idx]
    else:
        flight_time = None
    
    # 3. Find peak power (Power = Force × Velocity)
    power_vertical = vertical_force * vertical_vel
    peak_power = np.max(power_vertical)
    peak_power_per_kg = peak_power / body_mass
    
    print("=== CORRECTED JUMP ANALYSIS ===")
    print(f"Jump height: {jump_height*1000:.1f} mm ({jump_height:.3f} m)")
    print(f"Peak upward velocity: {peak_velocity:.2f} m/s")
    print(f"Peak vertical force: {peak_force:.0f} N ({peak_force_bw:.1f} BW)")
    print(f"Peak power: {peak_power:.0f} W ({peak_power_per_kg:.1f} W/kg)")
    
    if flight_time is not None:
        print(f"Flight time: {flight_time:.3f} s")
        if takeoff_idx is not None:
            print(f"Takeoff time: {time[takeoff_idx]:.2f} s")
        if landing_idx is not None:
            print(f"Landing time: {time[landing_idx]:.2f} s")
    
    # Validation checks
    print("\n=== VALIDATION ===")
    if 2 <= peak_force_bw <= 5:
        print("✓ Peak force looks realistic")
    else:
        print("⚠ Peak force still seems unusual")
    
    if 100 <= jump_height*1000 <= 600:
        print("✓ Jump height looks realistic")
    else:
        print("⚠ Jump height seems unusual")
    
    if 1.5 <= peak_velocity <= 4.0:
        print("✓ Peak velocity looks realistic")
    else:
        print("⚠ Peak velocity seems unusual")
    
    return {
        'jump_height': jump_height,
        'peak_velocity': peak_velocity,
        'peak_force': peak_force,
        'peak_force_bw': peak_force_bw,
        'peak_power': peak_power,
        'flight_time': flight_time,
        'takeoff_idx': takeoff_idx,
        'landing_idx': landing_idx
    }

def plot_corrected_analysis(results, metrics, body_mass, fs):
    """
    Plot the corrected analysis
    """
    r_com_raw = results["r_com_raw"] / 1000.0
    r_com = results["r_com"] / 1000.0
    v_com = results["v_com"] / 1000.0
    F_ext = results["F_ext_smooth"]
    
    time = np.arange(len(r_com)) / fs
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Corrected COM Analysis with Filtering', fontsize=16)
    
    # 1. Raw vs Filtered Position
    axes[0,0].plot(time, r_com_raw[:, 2], 'r-', alpha=0.5, label='Raw', linewidth=1)
    axes[0,0].plot(time, r_com[:, 2], 'b-', label='Filtered', linewidth=2)
    axes[0,0].set_ylabel('Vertical Position (m)')
    axes[0,0].set_title('COM Position: Raw vs Filtered')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Velocity
    axes[0,1].plot(time, v_com[:, 2], 'g-', linewidth=2)
    axes[0,1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    if metrics['takeoff_idx'] is not None:
        axes[0,1].axvline(time[metrics['takeoff_idx']], color='r', linestyle='--', alpha=0.7, label='Takeoff')
    if metrics['landing_idx'] is not None:
        axes[0,1].axvline(time[metrics['landing_idx']], color='orange', linestyle='--', alpha=0.7, label='Landing')
    axes[0,1].set_ylabel('Vertical Velocity (m/s)')
    axes[0,1].set_title('Vertical Velocity')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Force
    axes[1,0].plot(time, F_ext[:, 2], 'purple', linewidth=2)
    axes[1,0].axhline(y=body_mass*9.81, color='k', linestyle='--', alpha=0.5, label='Body weight')
    axes[1,0].set_ylabel('Vertical Force (N)')
    axes[1,0].set_xlabel('Time (s)')
    axes[1,0].set_title('Vertical Ground Reaction Force')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Power
    power = F_ext[:, 2] * v_com[:, 2]
    axes[1,1].plot(time, power, 'orange', linewidth=2)
    axes[1,1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1,1].set_ylabel('Power (W)')
    axes[1,1].set_xlabel('Time (s)')
    axes[1,1].set_title('Vertical Power')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Run the corrected analysis
print("Running corrected COM analysis with filtering...")
corrected_results = compute_whole_body_com(positions, body_mass, fs, cutoff_freq)

print("\nAnalyzing jump performance...")
jump_metrics = analyze_jump_performance(corrected_results, body_mass, fs)

print("\nCreating plots...")
plot_corrected_analysis(corrected_results, jump_metrics, body_mass, fs)