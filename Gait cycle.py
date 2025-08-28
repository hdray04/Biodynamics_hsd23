import ezc3d
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

DEBUG_DETECTOR = False  # basic mode


# --- Load cmj_1 only ---
walk_slow = ezc3d.c3d("/Users/harrietdray/Biodynamics/Harriet_c3d/Walk-001/pose_filt_0.c3d")
walk_fast = ezc3d.c3d("/Users/harrietdray/Biodynamics/Harriet_c3d/Walk-002/pose_filt_0.c3d")
jog = ezc3d.c3d("/Users/harrietdray/Biodynamics/Harriet_c3d/Jog-001/pose_filt_0.c3d")
labels = walk_fast['parameters']['POINT']['LABELS']['value']
labels_rotation = walk_fast['parameters']['ROTATION']['LABELS']['value']


def extract_angles(walk_fast, labels):
    angle_data = walk_fast['data']['points']
    n_frames = angle_data.shape[2]

    angle_indices = {
        'left_knee': labels.index('LeftKneeAngles_Theia'),
        'right_knee': labels.index('RightKneeAngles_Theia'),
        'left_hip': labels.index('LeftHipAngles_Theia'),
        'right_hip': labels.index('RightHipAngles_Theia'),
        'left_ankle': labels.index('LeftAnkleAngles_Theia'),
        'right_ankle': labels.index('RightAnkleAngles_Theia')
    }

    # Component 0 = flexion/extension for these Theia angles
    left_knee_flex = angle_data[0, angle_indices['left_knee'], :n_frames]
    right_knee_flex = angle_data[0, angle_indices['right_knee'], :n_frames]
    left_hip_flex = angle_data[0, angle_indices['left_hip'], :n_frames]
    right_hip_flex = angle_data[0, angle_indices['right_hip'], :n_frames]
    left_ankle_dorsi = angle_data[0, angle_indices['left_ankle'], :n_frames]
    right_ankle_dorsi = angle_data[0, angle_indices['right_ankle'], :n_frames]

    return {
        'knee': {'left': left_knee_flex, 'right': right_knee_flex},
        'hip': {'left': left_hip_flex, 'right': right_hip_flex},
        'ankle': {'left': left_ankle_dorsi, 'right': right_ankle_dorsi},
        'frames': np.arange(n_frames),
        'n_frames': n_frames
    }


# Extract 3D angles for all trials
angles = extract_angles(walk_fast, labels)

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

# --- Utilities ---

def fill_nans_1d(x):
    x = np.asarray(x, dtype=float)
    if not np.any(~np.isfinite(x)):
        return x
    n = x.size
    idx = np.arange(n)
    mask = np.isfinite(x)
    # If all are NaN, return zeros
    if mask.sum() == 0:
        return np.zeros_like(x)
    # For NaNs at the ends, extend edge values
    first = np.argmax(mask)
    last = n - 1 - np.argmax(mask[::-1])
    x[:first] = x[first]
    x[last+1:] = x[last]
    # Interpolate interior NaNs
    bad = ~mask
    x[bad] = np.interp(idx[bad], idx[mask], x[mask])
    return x

# --- Sanitize angle time series to avoid NaN-only segments ---

def sanitize_angles(angles):
    out = dict(angles)
    for joint in ('hip', 'knee', 'ankle'):
        for leg in ('left', 'right'):
            y = np.asarray(out[joint][leg], dtype=float)
            y = fill_nans_1d(y)
            out[joint][leg] = y
    return out

angles = sanitize_angles(angles)

# ===== BASIC GAIT EVENTS ONLY =====
# Simple smoothing, HS = minima of foot Z. TO = first upward vel zero-crossing of toe Z after each HS.

def detect_hs_basic(foot_z, fs, prominence=None, min_distance_s=None):
    """Basic heel-strike detection with light smoothing and adaptive thresholds.
    HS = minima of smoothed foot Z.
    Returns (idx, t[idx], z[idx]).
    """
    foot_z = np.asarray(foot_z)
    foot_z = np.where(np.isfinite(foot_z), foot_z, np.nanmedian(foot_z))
    # Smooth
    b, a = signal.butter(3, 6.0/(fs/2.0), 'low')
    z = signal.filtfilt(b, a, foot_z)

    # Adaptive params
    if prominence is None:
        p5, p95 = np.nanpercentile(z, [5, 95])
        amp = max(1e-6, float(p95 - p5))
        prominence = 0.15 * amp  # 15% of robust range
    if min_distance_s is None:
        # Estimate step frequency if possible
        f0 = estimate_step_frequency(z - np.mean(z), fs, band=(0.6, 3.0))
        if f0 is not None and f0 > 0:
            min_distance_s = max(0.2, 0.45 / f0)
        else:
            min_distance_s = 0.30

    min_dist = int(max(1, round(min_distance_s * fs)))

    # Primary detection on minima
    idx, _ = signal.find_peaks(-z, prominence=prominence, distance=min_dist)

    # Fallback 1: relax thresholds
    if idx.size == 0:
        idx, _ = signal.find_peaks(-z, prominence=0.5*prominence, distance=max(1, int(0.7*min_dist)))

    # Fallback 2: take deepest local minima by percentile threshold
    if idx.size == 0:
        minima = signal.argrelmin(z)[0]
        if minima.size:
            thr = np.nanpercentile(z[minima], 30)
            idx = minima[z[minima] <= thr]

    idx = np.asarray(idx, dtype=int)
    t = np.arange(len(z))/fs
    return idx, t[idx], z[idx]



def detect_to_basic(toe_z, fs, hs_idx, min_distance_s=0.2):
    toe_z = np.asarray(toe_z)
    toe_z = np.where(np.isfinite(toe_z), toe_z, np.nanmedian(toe_z))
    # Smooth
    b, a = signal.butter(3, 6.0/(fs/2.0), 'low')
    z = signal.filtfilt(b, a, toe_z)
    # Vertical velocity
    vel = np.gradient(z, 1.0/fs)
    # Upward zero-crossings
    up = np.where((vel[:-1] <= 0) & (vel[1:] > 0))[0] + 1
    # Enforce spacing
    min_dist = int(max(1, round(min_distance_s*fs)))
    if up.size > 1:
        keep = [up[0]]
        for k in up[1:]:
            if k - keep[-1] >= min_dist:
                keep.append(k)
        up = np.asarray(keep, int)
    # First zero-crossing after each HS and before next HS
    to_idx = []
    for i, s in enumerate(hs_idx):
        e = hs_idx[i+1] if i+1 < len(hs_idx) else len(z)
        cand = up[(up >= s) & (up < e)]
        if cand.size:
            to_idx.append(int(cand[0]))
    to_idx = np.asarray(to_idx, int)
    t = np.arange(len(z))/fs
    return to_idx, t[to_idx], z[to_idx]

# --- Cycle selection and improved toe-off ---

def best_cycle_by_step_time(hs_idx, fs):
    """Return (start,end) for the consecutive HS pair whose duration
    is closest to the median step time. Falls back to first pair.
    """
    hs_idx = np.asarray(hs_idx, int)
    if hs_idx.size < 2:
        return None
    d = np.diff(hs_idx)
    if d.size == 0:
        return int(hs_idx[0]), int(hs_idx[1])
    med = np.median(d)
    k = int(np.argmin(np.abs(d - med)))
    return int(hs_idx[k]), int(hs_idx[k+1])


def detect_to_threshold(toe_z, fs, hs_idx, rise_mm=15, hold_s=0.1, min_pct=0.45, max_pct=0.75):
    """Toe-off = first time toe height rises by rise_mm above a local baseline
    after HS and stays above for hold_s.
    """
    toe_z = np.asarray(toe_z, float)
    b, a = signal.butter(3, 6.0/(fs/2.0), 'low')
    z = signal.filtfilt(b, a, toe_z)
    hold_n = max(1, int(round(hold_s*fs)))
    to_idx = []
    for i, s in enumerate(hs_idx):
        e = hs_idx[i+1] if i+1 < len(hs_idx) else len(z)-1
        # baseline: 100 ms window after HS, clipped to segment
        b0 = max(s, min(e, s + int(0.1*fs)))
        base = np.nanmedian(z[s:b0]) if b0 > s else z[s]
        thr = base + rise_mm
        seg = z[s:e]
        if seg.size <= hold_n:
            continue
        above = seg > thr
        # find first index with a run of length hold_n
        run = 0
        found = None
        for j, flag in enumerate(above):
            run = run + 1 if flag else 0
            if run >= hold_n:
                found = s + j - hold_n + 1
                break
        if found is not None:
            min_i = int(s + min_pct * (e - s))
            max_i = int(s + max_pct * (e - s))
            if min_i <= found <= max_i:
                to_idx.append(int(found))
    return np.asarray(to_idx, int)


def plot_events_basic(time, l_foot_z, r_foot_z, l_hs, r_hs, l_to, r_to):
    fig, ax = plt.subplots(figsize=(11,4))
    ax.plot(time, l_foot_z, label='Left foot Z')
    ax.plot(time, r_foot_z, label='Right foot Z')
    if len(l_hs):
        ax.scatter(time[l_hs], l_foot_z[l_hs], marker='v', label='L HS')
    if len(r_hs):
        ax.scatter(time[r_hs], r_foot_z[r_hs], marker='v', label='R HS')
    if len(l_to):
        ax.scatter(time[l_to], l_foot_z[l_to], marker='^', label='L TO')
    if len(r_to):
        ax.scatter(time[r_to], r_foot_z[r_to], marker='^', label='R TO')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Height (mm)'); ax.set_title('Basic gait events')
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend()
    ax.grid(True, alpha=0.3)
    return fig

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

def detect_heel_strikes_detrended(
    foot_height,
    fs,
    min_prominence=8,
    min_distance_s=0.3,
    trend_cutoff_hz=0.6,
    auto_tune=True,
    prom_quantile_low=5,
    prom_quantile_high=95,
    prom_scale=0.25,
    step_band=(0.6, 3.0),   # Hz, typical step frequency range
    distance_factor=0.45     # fraction of step period used as min distance
):
    """Adaptive heel-strike detector.
    - Detrends foot Z then finds minima.
    - If auto_tune=True, it tunes prominence and min distance from signal stats and step frequency.
    Returns (idx, t[idx], h_at_idx, aux_dict) where aux_dict includes debug fields.
    """
    foot_height = np.asarray(foot_height)

    # 1) Light smoothing to reduce jitter
    bw_cut = 6.0
    b1, a1 = signal.butter(4, bw_cut / (fs / 2.0), 'low')
    z_smooth = signal.filtfilt(b1, a1, foot_height)

    # 2) Estimate slow baseline trend and remove it
    b2, a2 = signal.butter(2, trend_cutoff_hz / (fs / 2.0), 'low')
    trend = signal.filtfilt(b2, a2, z_smooth)
    osc = z_smooth - trend

    tuned = {
        'trend_cutoff_hz': trend_cutoff_hz,
        'min_prominence': float(min_prominence),
        'min_distance_s': float(min_distance_s),
        'distance_samples': int(max(1, round(min_distance_s * fs))),
    }

    # 3) Auto-tune from signal if requested
    if auto_tune and np.any(np.isfinite(osc)):
        # Estimate step frequency from the detrended oscillation
        f0 = estimate_step_frequency(osc, fs, band=step_band)
        if f0 is not None and f0 > 0:
            # Set min distance as a fraction of step period
            min_distance_s = max(0.2, distance_factor / float(f0))
            tuned['estimated_step_hz'] = float(f0)
            tuned['min_distance_s'] = float(min_distance_s)
            tuned['distance_samples'] = int(max(1, round(min_distance_s * fs)))

            # If trend cutoff is too low relative to step freq, raise it slightly
            # Keep it safely below the step band
            new_trend = min(0.5 * f0, 0.8)
            if new_trend > trend_cutoff_hz:
                trend_cutoff_hz = new_trend
                b2, a2 = signal.butter(2, trend_cutoff_hz / (fs / 2.0), 'low')
                trend = signal.filtfilt(b2, a2, z_smooth)
                osc = z_smooth - trend
                tuned['trend_cutoff_hz'] = float(trend_cutoff_hz)

        # Robust amplitude scale using inter-quantile range
        lo = np.nanpercentile(osc, prom_quantile_low)
        hi = np.nanpercentile(osc, prom_quantile_high)
        iqr_amp = max(1e-6, hi - lo)
        auto_prom = prom_scale * iqr_amp
        if auto_prom > min_prominence:
            min_prominence = float(auto_prom)
            tuned['min_prominence'] = float(min_prominence)
            tuned['prom_quantiles'] = (float(lo), float(hi))

    # 4) Find minima on negative oscillation
    min_distance = int(max(1, round(min_distance_s * fs)))
    idx, props = signal.find_peaks(-osc, prominence=min_prominence, distance=min_distance)

    t = np.arange(len(foot_height)) / fs
    h = foot_height[idx]

    aux = {"trend": trend, "osc": osc, "z_smooth": z_smooth, "props": props, "tuned": tuned}
    return idx, t[idx], h, aux

def estimate_step_frequency(sig, fs, band=(0.6, 3.0)):
    sig = np.asarray(sig)
    if sig.size < 10:
        return None
    # Remove NaNs
    sig = np.where(np.isfinite(sig), sig, 0.0)
    # Zero-mean
    sig = sig - np.mean(sig)
    # Power spectrum via rFFT
    fft = np.fft.rfft(sig)
    freqs = np.fft.rfftfreq(sig.size, d=1.0/fs)
    # Limit to band
    lo, hi = band
    band_mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(band_mask):
        return None
    power = np.abs(fft)**2
    peak_idx = np.argmax(power[band_mask])
    f_peak = freqs[band_mask][peak_idx]
    return float(f_peak)

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

# --- Helpers: ROM and cycle normalization ---

def compute_rom(sig):
    sig = np.asarray(sig)
    return float(np.nanmax(sig) - np.nanmin(sig))


def resample_cycle(sig, start_idx, end_idx, n_points=101):
    sig = np.asarray(sig)
    if end_idx <= start_idx:
        end_idx = start_idx + 1
    x = np.arange(start_idx, end_idx)
    y = sig[start_idx:end_idx]
    xp = np.linspace(start_idx, end_idx - 1, n_points)
    return np.interp(xp, x, y)

def knee_cycle_from_angles(angles, leg='left', prominence=5, min_distance_s=0.3, fs=100, invert=True):
    """Fallback: derive a single gait cycle (start,end) from knee angle peaks.
    Returns a tuple (start_idx, end_idx) or None if unavailable.
    """
    sig = np.asarray(angles['knee'][leg])
    if invert:
        sig = -sig  # make flexion positive if needed
    min_dist = int(max(1, round(min_distance_s * fs)))
    idx, _ = signal.find_peaks(sig, prominence=prominence, distance=min_dist)
    return (int(idx[0]), int(idx[1])) if idx.size >= 2 else None

# --- Generic plot over time with events for one joint ---

def plot_joint_over_time_with_events(angles, hs_idx, to_idx, sampling_rate=100, leg='left', joint='knee', invert=False):
    assert leg in ('left', 'right')
    assert joint in ('knee', 'hip', 'ankle')

    series = np.asarray(angles[joint][leg])
    if invert:
        series = -series

    t = np.arange(len(series)) / sampling_rate

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, series, label=f"{leg.capitalize()} {joint} angle")

    # Events
    hs_idx = np.asarray(hs_idx, dtype=int)
    to_idx = np.asarray(to_idx, dtype=int)
    hs_idx = hs_idx[(hs_idx >= 0) & (hs_idx < len(series))]
    to_idx = to_idx[(to_idx >= 0) & (to_idx < len(series))]

    if hs_idx.size:
        ax.scatter(t[hs_idx], series[hs_idx], marker='v', label='Heel strike', zorder=5)
    if to_idx.size:
        ax.scatter(t[to_idx], series[to_idx], marker='^', label='Toe-off', zorder=6)

    rom = compute_rom(series)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle (deg)')
    ax.set_title(f"{leg.capitalize()} {joint} over time  |  ROM = {rom:.1f}°")
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend()
    return fig

# --- Symmetry plot: left vs right in a single normalized gait cycle ---

def plot_symmetry_one_cycle(angles, hs_left_idx, hs_right_idx, joint='knee', sampling_rate=100, invert=False):
    assert joint in ('knee', 'hip', 'ankle')

    left = np.asarray(angles[joint]['left'])
    right = np.asarray(angles[joint]['right'])
    if invert:
        left, right = -left, -right

    # Choose first complete cycle for each leg based on heel strikes
    def first_cycle(hs_idx, length):
        hs_idx = np.asarray(hs_idx, dtype=int)
        hs_idx = hs_idx[(hs_idx >= 0) & (hs_idx < length)]
        if hs_idx.size < 2:
            return None
        return int(hs_idx[0]), int(hs_idx[1])

    l_pair = first_cycle(hs_left_idx, len(left))
    r_pair = first_cycle(hs_right_idx, len(right))

    # Fallback: if heel-strike cycles are missing, try knee-angle based cycles
    if l_pair is None:
        l_pair = knee_cycle_from_angles(angles, 'left', fs=sampling_rate, invert=invert)
    if r_pair is None:
        r_pair = knee_cycle_from_angles(angles, 'right', fs=sampling_rate, invert=invert)

    if l_pair is None and r_pair is None:
        print(f"[symmetry:{joint}] No heel-strike cycles and knee-angle fallback failed.")
    elif (l_pair is None) or (r_pair is None):
        print(f"[symmetry:{joint}] Using knee-angle fallback for one leg.")

    fig, ax = plt.subplots(figsize=(10, 4))
    x_pct = np.linspace(0, 100, 101)

    have_any = False
    if l_pair:
        l_cycle = resample_cycle(left, l_pair[0], l_pair[1])
        ax.plot(x_pct, l_cycle, label='Left')
        have_any = True
    if r_pair:
        r_cycle = resample_cycle(right, r_pair[0], r_pair[1])
        ax.plot(x_pct, r_cycle, label='Right')
        have_any = True

    # Simple symmetry metrics
    if l_pair and r_pair:
        si_mean = float(np.nanmean(l_cycle - r_cycle))
        si_rmse = float(np.sqrt(np.nanmean((l_cycle - r_cycle)**2)))
        title_tail = f" | Δmean={si_mean:.1f}°, RMSE={si_rmse:.1f}°"
    else:
        title_tail = ""

    ax.set_xlabel('Gait cycle (%)')
    ax.set_ylabel('Angle (deg)')

    if have_any:
        ax.set_title(f'{joint.capitalize()} symmetry (Left vs Right){title_tail}')
        ax.grid(True, alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend()
    else:
        ax.set_title(f'{joint.capitalize()} symmetry (Left vs Right)')
        ax.text(0.5, 0.5, 'No complete gait cycle detected for plotting', ha='center', va='center', transform=ax.transAxes)
        ax.grid(True, alpha=0.3)

    return fig


def plot_knee_flexion_with_events(angles, hs_idx, to_idx, sampling_rate = 100, leg = 'left'):
    assert leg in ('left', 'right')
    knee_flexion = angles['knee'][leg]
    knee_flexion = -knee_flexion
    n = len(knee_flexion)
    t = np.arange(n) / sampling_rate

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(t, knee_flexion, label=f'{leg.capitalize()} knee flexion')

    ax.scatter(t[np.asarray(hs_idx, int)], knee_flexion[np.asarray(hs_idx, int)],
               marker='v', label='Heel strike', zorder=5)
    ax.scatter(t[np.asarray(to_idx, int)], knee_flexion[np.asarray(to_idx, int)],
               marker='^', label='Toe-off', zorder=6)

    rom = compute_rom(knee_flexion)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Knee Flexion (deg)')
    ax.set_title(f'{leg.capitalize()} knee flexion with gait events  |  ROM = {rom:.1f}°')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


# --- One-cycle normalization and plotting ---

def first_complete_cycle(hs_idx, n):
    hs_idx = np.asarray(hs_idx, dtype=int)
    hs_idx = hs_idx[(hs_idx >= 0) & (hs_idx < n)]
    if hs_idx.size < 2:
        return None
    return int(hs_idx[0]), int(hs_idx[1])


def normalize_one_cycle(angles, leg, start_idx, end_idx):
    # light low-pass on angles
    def lp(y):
        y = np.asarray(y, float)
        b, a = signal.butter(3, 6.0/(100/2.0), 'low')  # assumes 100 Hz
        return signal.filtfilt(b, a, y)

    # Guard against too-short windows
    if end_idx - start_idx < 5:
        end_idx = start_idx + 5

    hip = resample_cycle(lp(angles['hip'][leg]), start_idx, end_idx)
    knee = resample_cycle(lp(-angles['knee'][leg]), start_idx, end_idx)   # flexion positive
    if leg == 'right':
        knee = -knee  # flip right knee to make flexion positive
    ankle = resample_cycle(lp(angles['ankle'][leg]), start_idx, end_idx)  # dorsiflexion positive
    x_pct = np.linspace(0, 100, len(hip))
    return x_pct, hip, knee, ankle


def plot_three_joint_cycle(angles, hs_idx, to_idx=None, leg='left', sampling_rate=100):
    n = angles['n_frames']
    hs_idx = np.asarray(hs_idx, int)
    if hs_idx.size < 2:
        print('[cycle] Need at least two heel strikes for the selected leg')
        return None
    pair = best_cycle_by_step_time(hs_idx, sampling_rate) or first_complete_cycle(hs_idx, n)
    if pair is None:
        print('[cycle] Could not select a valid cycle')
        return None
    s, e = pair
    x, hip, knee, ankle = normalize_one_cycle(angles, leg, s, e)

    # Check for non-finite or empty arrays
    if not np.any(np.isfinite(hip)) and not np.any(np.isfinite(knee)) and not np.any(np.isfinite(ankle)):
        print('[cycle] Angle arrays are not finite after sanitization')
        return None

    to_pct = None
    if to_idx is not None and len(to_idx):
        to_idx = np.asarray(to_idx, int)
        mask = (to_idx >= s) & (to_idx < e)
        if np.any(mask):
            first_to = int(to_idx[mask][0])
            to_pct = 100.0 * (first_to - s) / max(1, (e - s))

    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    data = [('Hip', hip), ('Knee', knee), ('Ankle', ankle)]
    for ax, (title, y) in zip(axes, data):
        ax.plot(x, y)
        ax.axhline(0.0, linestyle='--', linewidth=1)
        if to_pct is not None:
            ax.axvline(to_pct, linestyle='--', linewidth=1)
        ax.set_title(title)
        ax.set_ylabel('Degrees')
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Gait Cycle (100%)')
    plt.tight_layout()
    return fig


# --- Overlay left and right (one cycle each) ---
def plot_three_joint_cycle_both(angles, l_hs, r_hs, l_to=None, r_to=None, sampling_rate=100):
    n = angles['n_frames']
    # Select cycles
    def pick(hs):
        hs = np.asarray(hs, int)
        if hs.size < 2:
            return None
        return best_cycle_by_step_time(hs, sampling_rate) or first_complete_cycle(hs, n)

    l_pair = pick(l_hs)
    r_pair = pick(r_hs)
    if l_pair is None and r_pair is None:
        print('[overlay] Need at least two heel strikes for one leg')
        return None

    # Normalize for each leg if available
    data = {}
    if l_pair is not None:
        ls, le = l_pair
        xL, hipL, kneeL, ankleL = normalize_one_cycle(angles, 'left', ls, le)
        data['L'] = (xL, hipL, kneeL, ankleL, ls, le)
    if r_pair is not None:
        rs, re = r_pair
        xR, hipR, kneeR, ankleR = normalize_one_cycle(angles, 'right', rs, re)
        data['R'] = (xR, hipR, kneeR, ankleR, rs, re)

    # Toe-off percentages
    def to_pct(to_idx, s, e):
        if to_idx is None:
            return None
        to_idx = np.asarray(to_idx, int)
        mask = (to_idx >= s) & (to_idx < e)
        if np.any(mask):
            first_to = int(to_idx[mask][0])
            return 100.0 * (first_to - s) / max(1, (e - s))
        return None

    toL = to_pct(l_to, l_pair[0], l_pair[1]) if l_pair is not None else None
    toR = to_pct(r_to, r_pair[0], r_pair[1]) if r_pair is not None else None

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    joints = ['Hip', 'Knee', 'Ankle']
    for ax, j in zip(axes, joints):
        # Left
        if 'L' in data:
            x, hip, knee, ankle, _, _ = data['L']
            y = {'Hip': hip, 'Knee': knee, 'Ankle': ankle}[j]
            ax.plot(x, y, label='Left')
        # Right
        if 'R' in data:
            x, hip, knee, ankle, _, _ = data['R']
            y = {'Hip': hip, 'Knee': knee, 'Ankle': ankle}[j]
            ax.plot(x, y, label='Right')
        ax.axhline(0.0, linestyle='--', linewidth=1)
        if toL is not None:
            ax.axvline(toL, linestyle='--', linewidth=1, label='L TO' if j == 'Hip' else None)
        if toR is not None:
            ax.axvline(toR, linestyle=':', linewidth=1, label='R TO' if j == 'Hip' else None)
        ax.set_title(j)
        ax.set_ylabel('Degrees')
        ax.grid(True, alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend()
    axes[-1].set_xlabel('Gait Cycle (100%)')
    plt.tight_layout()
    return fig

def draw_detector_debug(title, t, raw, aux, idx):
    if not DEBUG_DETECTOR:
        return
    trend = aux.get('trend')
    osc = aux.get('osc')
    z_smooth = aux.get('z_smooth')
    tuned = aux.get('tuned', {})
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(t, raw, label='raw Z')
    axes[0].plot(t, z_smooth, label='smooth')
    axes[0].plot(t, trend, label='trend')
    axes[0].legend(); axes[0].set_title(f"{title} | trend_cutoff={tuned.get('trend_cutoff_hz'):.2f} Hz")
    axes[1].plot(t, z_smooth - trend, label='osc')
    axes[1].scatter(t[idx], (z_smooth - trend)[idx], c='k', s=20, label='HS minima')
    axes[1].legend(); axes[1].set_ylabel('osc (mm)')
    axes[2].plot(t, raw, label='raw')
    axes[2].scatter(t[idx], raw[idx], c='r', s=20, label='HS on raw')
    axes[2].legend(); axes[2].set_xlabel('Time (s)')
    axes[2].text(0.01, 0.6, str(tuned), transform=axes[2].transAxes, fontsize=8, va='top')
    plt.tight_layout()

# --- BASIC MODE: events only ---
fs = 100
l_foot_z = positions['l_foot'][:, 2]
r_foot_z = positions['r_foot'][:, 2]
l_toe_z  = positions['l_toes'][:, 2]
r_toe_z  = positions['r_toes'][:, 2]

# Sanitize NaNs before any processing
l_foot_z = fill_nans_1d(l_foot_z)
r_foot_z = fill_nans_1d(r_foot_z)
l_toe_z  = fill_nans_1d(l_toe_z)
r_toe_z  = fill_nans_1d(r_toe_z)

n = len(l_foot_z)
time = np.arange(n)/fs

# Print Z range stats for quick scale check
print(f"Z ranges (mm): L[{np.nanmin(l_foot_z):.1f},{np.nanmax(l_foot_z):.1f}] R[{np.nanmin(r_foot_z):.1f},{np.nanmax(r_foot_z):.1f}]  fs={fs}")

# Heel strikes
l_hs, _, _ = detect_hs_basic(l_foot_z, fs, prominence=None, min_distance_s=None)
r_hs, _, _ = detect_hs_basic(r_foot_z, fs, prominence=None, min_distance_s=None)

# Toe-offs
l_to, _, _ = detect_to_basic(l_toe_z, fs, l_hs, min_distance_s=0.2)
r_to, _, _ = detect_to_basic(r_toe_z, fs, r_hs, min_distance_s=0.2)

# Improved toe-off by sustained rise
l_to_thr = detect_to_threshold(l_toe_z, fs, l_hs, rise_mm=15, hold_s=0.1)
r_to_thr = detect_to_threshold(r_toe_z, fs, r_hs, rise_mm=15, hold_s=0.1)
# Prefer threshold detector if it found events
if l_to_thr.size:
    l_to = l_to_thr
if r_to_thr.size:
    r_to = r_to_thr

print(f"L HS={len(l_hs)}, R HS={len(r_hs)}, L TO={len(l_to)}, R TO={len(r_to)}")


plot_events_basic(time, l_foot_z, r_foot_z, l_hs, r_hs, l_to, r_to)

# --- Produce one-cycle hip/knee/ankle plot (LEFT leg) ---
if len(l_hs) >= 2:
    s_dbg, e_dbg = best_cycle_by_step_time(l_hs, fs)
    print(f"[cycle] Left HS pair used: start={s_dbg}, end={e_dbg}, length={e_dbg - s_dbg} frames")
plot_three_joint_cycle(angles, l_hs, to_idx=l_to, leg='left', sampling_rate=fs)


# --- Overlay left and right (one cycle each) ---
plot_three_joint_cycle_both(angles, l_hs, r_hs, l_to=l_to, r_to=r_to, sampling_rate=fs)

plt.show()
