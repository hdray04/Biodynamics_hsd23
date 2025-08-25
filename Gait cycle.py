import ezc3d
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


# --- Load cmj_1 only ---
walk_slow = ezc3d.c3d("/Users/harrietdray/Biodynamics/Harriet_c3d/Walk-001/pose_filt_0.c3d")
walk_fast = ezc3d.c3d("/Users/harrietdray/Biodynamics/Harriet_c3d/Walk-002/pose_filt_0.c3d")
jog = ezc3d.c3d("/Users/harrietdray/Biodynamics/Harriet_c3d/Jog-001/pose_filt_0.c3d")
labels_rotation = walk_fast['parameters']['ROTATION']['LABELS']['value']


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

rotation_data_1 = walk_fast['data']['rotations']
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
    """
    Parameters
    ----------
    foot_height : 1D array (mm)
        Vertical position of the foot over time (per frame).
    fs : float
        Sampling rate (Hz).
    min_prominence : float (mm)
        Minimum prominence for a minimum to be counted (filters tiny wiggles).
    min_distance_s : float (s)
        Minimum time between consecutive heel strikes (controls cadence; 0.4s ~ 150 steps/min).
    trend_cutoff_hz : float (Hz)
        Low cutoff for baseline trend. Must be < step frequency (e.g., 0.2–0.5 Hz).

    Returns
    -------
    idx : ndarray of int
        Indices of detected heel strikes (minima) in the original signal.
    t : ndarray of float
        Times (s) of the heel strikes.
    h : ndarray of float
        Heights (mm) at the heel strikes in the original signal.
    aux : dict
        Contains 'trend' (baseline) and 'osc' (detrended oscillation) for debugging/plotting.
    """
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

def plot_with_heel_strikes(positions, sampling_rate=100):
    # Get basic foot height data
    fig, ax, l_foot_height, r_foot_height, time_axis = foot_height_over_time(positions, sampling_rate)
    
    # Detect heel strikes
    l_strikes_idx, l_strikes_time, l_strikes_height, l_aux = detect_heel_strikes_detrended(l_foot_height, sampling_rate)
    r_strikes_idx, r_strikes_time, r_strikes_height, r_aux = detect_heel_strikes_detrended(r_foot_height, sampling_rate)
    
    
   
   #Optional: draw the slow baseline (faint) for diagnostics
    ax.plot(time_axis, l_aux["trend"], linewidth=1, alpha=0.3)
    ax.plot(time_axis, r_aux["trend"], linewidth=1, alpha=0.3)

   #Overlay heel strikes 
    ax.scatter(l_strikes_time, l_strikes_height, zorder=5, label='Left Heel Strikes')
    ax.scatter(r_strikes_time, r_strikes_height, zorder=5, label='Right Heel Strikes')
    ax.legend()

    print(f"\nHeel Strikes Detected:")
    print(f"Left foot: {len(l_strikes_time)} heel strikes")
    print(f"Right foot: {len(r_strikes_time)} heel strikes")
    
    if len(l_strikes_time) > 1:
        l_step_times = np.diff(l_strikes_time)
        print(f"Left step duration: {np.mean(l_step_times):.2f} ± {np.std(l_step_times):.2f} seconds")
    
    if len(r_strikes_time) > 1:
        r_step_times = np.diff(r_strikes_time)
        print(f"Right step duration: {np.mean(r_step_times):.2f} ± {np.std(r_step_times):.2f} seconds")
    
    return fig, (l_strikes_idx, r_strikes_idx)

fig, (l_idx, r_idx) = plot_with_heel_strikes(positions, sampling_rate = 100) 
plt.show()
