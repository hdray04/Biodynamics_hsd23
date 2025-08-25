import ezc3d
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


# --- Load cmj_1 only ---
walk_slow = ezc3d.c3d("/Users/harrietdray/Biodynamics/Harriet_c3d/Walk-001/pose_filt_0.c3d")
walk_fast = ezc3d.c3d("/Users/harrietdray/Biodynamics/Harriet_c3d/Walk-002/pose_filt_0.c3d")
jog = ezc3d.c3d("/Users/harrietdray/Biodynamics/Harriet_c3d/Jog-001/pose_filt_0.c3d")
labels = walk_slow['parameters']['POINT']['LABELS']['value']
labels_rotation = walk_slow['parameters']['ROTATION']['LABELS']['value']


def extract_angles(walk_slow, labels):
    angle_data_trials = [walk_slow['data']['points']]
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

    angle_data = angle_data_trials[0]
    n_frames = angle_data.shape[2]
    
    left_knee_flexion = angle_data[0, angle_indices['left_knee'], :n_frames]
    right_knee_flexion = angle_data[0, angle_indices['right_knee'], :n_frames]

    return {
        'left': left_knee_flexion,
        'right': right_knee_flexion,
        'frames': np.arange(n_frames),
        'n_frames': n_frames
    }


# Extract 3D angles for all trials
angles = extract_angles(walk_slow, labels)

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

rotation_data_1 = walk_slow['data']['rotations']
matrices = extract_matrices(rotation_data_1, labels_rotation)

# --- Extract positions from matrices ---
def extract_positions_from_matrices(matrices):
    positions_dict = {}
    for joint_name, joint_matrices in matrices.items():
        positions = joint_matrices[:, :3, 3]  # (frames, 3)
        positions_dict[joint_name] = positions
    return positions_dict

positions = extract_positions_from_matrices(matrices)

def foot_height_over_time(positions, sampling_rate=100):
    l_foot = positions['l_foot']
    r_foot = positions['r_foot']
    l_foot_height = l_foot[:, 2]  # z-coordinates for left foot (mm)
    r_foot_height = r_foot[:, 2]  # z-coordinates for right foot (mm)

    n_frames = len(l_foot_height)
    time = np.arange(n_frames) / sampling_rate  # Time vector in seconds

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time, l_foot_height, label='Left Foot')
    ax.plot(time, r_foot_height, label='Right Foot')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Foot Height (mm)')
    ax.set_title('Foot Height Over Time')
    ax.legend()
    ax.grid(True)

    # IMPORTANT: return 5 values
    return fig, ax, l_foot_height, r_foot_height, time


def detect_simple_heel_strikes(foot_height, time_axis, min_height_threshold=None):
    from scipy import signal
    
    if min_height_threshold is None:
        min_height_threshold = np.min(foot_height) + 10  # 10mm above minimum
    
    # Find local minima that are close to ground level
    # Use prominence to avoid tiny fluctuations
    minima_indices, properties = signal.find_peaks(-foot_height, 
                                                  height=-min_height_threshold,
                                                  prominence=20,  # Adjust based on your data
                                                  distance=50)    # Min 0.5s between steps at 100Hz
    
    heel_strike_times = time_axis[minima_indices]
    heel_strike_heights = foot_height[minima_indices]
    
    return minima_indices, heel_strike_times, heel_strike_heights

# Robust heel-strike detection: remove slow drift, then find local minima per cycle
# This addresses cases where foot Z keeps slowly increasing so only the last minimum is detected.

def detect_heel_strikes_detrended(foot_height, fs, min_prominence=15, min_distance_s=0.4, trend_cutoff_hz=0.3):

    foot_height = np.asarray(foot_height)

    # 1) Light smoothing at movement bandwidth to reduce frame-to-frame jitter
    bw_cut = 6.0
    b1, a1 = signal.butter(4, bw_cut / (fs / 2.0), 'low')
    z_smooth = signal.filtfilt(b1, a1, foot_height)

    # 2) Estimate slow baseline trend and remove it
    #    trend_cutoff_hz should be << step frequency so it captures only drift, not steps
    b2, a2 = signal.butter(2, trend_cutoff_hz / (fs / 2.0), 'low')
    trend = signal.filtfilt(b2, a2, z_smooth)
    osc = z_smooth - trend

    # 3) Find minima of the oscillation using find_peaks on the negative signal
    min_distance = int(max(1, round(min_distance_s * fs)))
    idx, props = signal.find_peaks(-osc, prominence=min_prominence, distance=min_distance)

    t = np.arange(len(foot_height)) / fs
    h = foot_height[idx]
    return idx, t[idx], h, {"trend": trend, "osc": osc}

# Toe-off detection: use detrended toe height and vertical velocity zero-crossings
# Picks the first upward zero-crossing after each heel strike and before the next heel strike

def detect_toe_off_detrended(toe_height, fs, heel_strike_idx, trend_cutoff_hz=0.3, ground_window_mm=20, min_distance_s=0.2, sampling_rate=100):

    toe_height = np.asarray(toe_height)

    # Smooth
    bw_cut = 6.0
    b1, a1 = signal.butter(4, bw_cut / (fs / 2.0), 'low')
    z_smooth = signal.filtfilt(b1, a1, toe_height)

    # Baseline trend and oscillation
    b2, a2 = signal.butter(2, trend_cutoff_hz / (fs / 2.0), 'low')
    trend = signal.filtfilt(b2, a2, z_smooth)
    osc = z_smooth - trend

    # Vertical velocity
    vel = np.gradient(z_smooth, 1.0 / fs)

    # Upward zero-crossings of velocity: vel crosses from <=0 to >0
    sign_prev = vel[:-1] <= 0
    sign_next = vel[1:] > 0
    zc_idx = np.where(sign_prev & sign_next)[0] + 1  # indices where crossing occurs

    # Keep only zero-crossings near ground level
    near_ground = np.abs(osc) <= ground_window_mm
    zc_idx = zc_idx[near_ground[zc_idx]]

    # Enforce a minimum distance constraint (cadence sanity)
    min_dist = int(max(1, round(min_distance_s * fs)))
    if zc_idx.size > 1:
        keep = [zc_idx[0]]
        for k in zc_idx[1:]:
            if k - keep[-1] >= min_dist:
                keep.append(k)
        zc_idx = np.asarray(keep, dtype=int)

    # Select the first valid zero-crossing after each heel strike and before the next heel strike
    toe_off_idx = []
    for i in range(len(heel_strike_idx)):
        start_i = heel_strike_idx[i]
        end_i = heel_strike_idx[i + 1] if i + 1 < len(heel_strike_idx) else len(toe_height)
        # candidate crossings in [start_i, end_i)
        cand = zc_idx[(zc_idx >= start_i) & (zc_idx < end_i)]
        if cand.size:
            toe_off_idx.append(int(cand[0]))

    toe_off_idx = np.asarray(toe_off_idx, dtype=int)
    t = np.arange(len(toe_height)) / fs
    h = toe_height[toe_off_idx]

    return toe_off_idx, t[toe_off_idx], h, {"trend": trend, "osc": osc, "vel": vel}

def plot_with_heel_strikes(positions, sampling_rate=100):
    # Get basic foot height data
    fig, ax, l_foot_height, r_foot_height, time_axis = foot_height_over_time(positions, sampling_rate)
    
    # Detect heel strikes
    l_strikes_idx, l_strikes_time, l_strikes_height, l_aux = detect_heel_strikes_detrended(l_foot_height, sampling_rate)
    r_strikes_idx, r_strikes_time, r_strikes_height, r_aux = detect_heel_strikes_detrended(r_foot_height, sampling_rate)

    # Get toe heights
    l_toe_height = positions['l_toes'][:, 2]
    r_toe_height = positions['r_toes'][:, 2]

    # Detect toe-offs using heel-strike windows
    l_toeoff_idx, l_toeoff_t, l_toeoff_h, l_toe_aux = detect_toe_off_detrended(l_toe_height, sampling_rate, l_strikes_idx)
    r_toeoff_idx, r_toeoff_t, r_toeoff_h, r_toe_aux = detect_toe_off_detrended(r_toe_height, sampling_rate, r_strikes_idx)
    
    
   
   #Optional: draw the slow baseline (faint) for diagnostics
    ax.plot(time_axis, l_aux["trend"], linewidth=1, alpha=0.3)
    ax.plot(time_axis, r_aux["trend"], linewidth=1, alpha=0.3)

   #Overlay heel strikes 
    ax.scatter(l_strikes_time, l_strikes_height, zorder=5, label='Left Heel Strikes')
    ax.scatter(r_strikes_time, r_strikes_height, zorder=5, label='Right Heel Strikes')

    # Overlay toe-offs
    ax.scatter(l_toeoff_t, l_toeoff_h, marker='^', zorder=6, label='Left Toe-Off')
    ax.scatter(r_toeoff_t, r_toeoff_h, marker='^', zorder=6, label='Right Toe-Off')

    ax.legend()

    print(f"\nHeel Strikes Detected:")
    print(f"Left foot: {len(l_strikes_time)} heel strikes")
    print(f"Right foot: {len(r_strikes_time)} heel strikes")
    print(f"Left foot: {len(l_toeoff_t)} toe-offs")
    print(f"Right foot: {len(r_toeoff_t)} toe-offs")
    
    if len(l_strikes_time) > 1:
        l_step_times = np.diff(l_strikes_time)
        print(f"Left step duration: {np.mean(l_step_times):.2f} ± {np.std(l_step_times):.2f} seconds")
    
    if len(r_strikes_time) > 1:
        r_step_times = np.diff(r_strikes_time)
        print(f"Right step duration: {np.mean(r_step_times):.2f} ± {np.std(r_step_times):.2f} seconds")
    
    return fig, (l_strikes_idx, r_strikes_idx), (l_toeoff_idx, r_toeoff_idx)


# --- New function: plot_gait_phases_one_leg ---
def plot_knee_flexion_with_events(angles, hs_idx, to_idx, sampling_rate = 100, leg = 'left'):
    assert leg in ('left', 'right')
    if leg == 'left':
        knee_flexion = angles['left']
    else:
        knee_flexion = angles['right']
    knee_flexion = -knee_flexion
    n = len(knee_flexion)
    t = np.arange(n) / sampling_rate

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(t, knee_flexion, label=f'{leg.capitalize()} knee flexion')

    # Overlay heel strikes
    ax.scatter(t[hs_idx], knee_flexion[hs_idx],
               marker='v', color='blue', label='Heel strike', zorder=5)
    # Overlay toe-offs
    ax.scatter(t[to_idx], knee_flexion[to_idx],
               marker='^', color='orange', label='Toe-off', zorder=6)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Knee Flexion (deg)')
    ax.set_title(f'{leg.capitalize()} knee flexion with gait events')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


# --- Use new function for simplified one-leg plot (left leg by default) ---
# Detect gait events for left leg
l_foot_height = positions['l_foot'][:, 2]
l_toe_height = positions['l_toes'][:, 2]
hs_idx, _, _, _ = detect_heel_strikes_detrended(l_foot_height, fs=100)
to_idx, _, _, _ = detect_toe_off_detrended(l_toe_height, fs=100, heel_strike_idx=hs_idx)
fig2 = plot_knee_flexion_with_events(angles, np.array(hs_idx), np.array(to_idx), sampling_rate=100, leg='left')
plt.show()
