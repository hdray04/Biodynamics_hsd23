import ezc3d
import numpy as np 
import matplotlib as plt 
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

"""
Previously this script combined CMJ trials. Per request, switch to
single-trial analysis: extract labels and rotation labels from cmj_1
and analyze only that trial.

Also used for single leg hops and drop jumps if modified accordingly.
"""

# Load only cmj_1
cmj_1 = ezc3d.c3d('/Users/harrietdray/Library/CloudStorage/OneDrive-ImperialCollegeLondon/ACL_data/Pilot - Tash/Pilot - Tash_c3d - Sorted/single_hop_left_2/pose_filt_0.c3d')

params = cmj_1['parameters']
fs = 100
time = np.arange(cmj_1['data']['points'].shape[2]) / fs

print("Param groups:", list(params.keys()))

def find_any_angle_fields(params):
    hits = []
    for grp_name, grp in params.items():
        for sub_name, sub in grp.items():
            try:
                if isinstance(sub, dict):
                    # label arrays
                    if 'value' in sub and isinstance(sub['value'], list):
                        vals = sub['value']
                        if any(isinstance(v, str) and 'angle' in v.lower() for v in vals):
                            hits.append((grp_name, sub_name, 'LABELS', vals[:10]))
                # group names with 'angle'
                if 'angle' in sub_name.lower() or 'angle' in grp_name.lower():
                    hits.append((grp_name, sub_name, 'GROUP_OR_FIELD', None))
            except Exception:
                pass
    return hits

hits = find_any_angle_fields(params)
print("Angle-like fields found:", hits)

# Also check ANALOG and ROTATION labels explicitly
for key in ['POINT', 'ANALOG', 'ROTATION']:
    if key in params and 'LABELS' in params[key]:
        print(f"{key}.LABELS count:", len(params[key]['LABELS']['value']))
        print(f"Sample {key} labels:", params[key]['LABELS']['value'][:20])

# Single-trial data arrays
angle_data = cmj_1['data']['points']
rotation_data = cmj_1['data']['rotations']

# Extract labels from cmj_1 only
labels = cmj_1['parameters']['POINT']['LABELS']['value']
labels_rotation = cmj_1['parameters']['ROTATION']['LABELS']['value']
units = cmj_1['parameters']['POINT']['UNITS']['value']


# Optional: quick diagnostics to check cmj_2 and cmj_3 label availability
RUN_DIAGNOSTICS = False

def _find_angle_like(params):
    hits = []
    for gname, group in params.items():
        for fname, field in group.items():
            try:
                vals = field.get('value', [])
                if isinstance(vals, list) and any(isinstance(v, str) and 'angle' in v.lower() for v in vals):
                    hits.append((gname, fname, vals[:10]))
            except Exception:
                pass
    return hits


def extract_3d_angles(angle_data, labels):
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

    n_frames = angle_data.shape[2]
    all_angles = {}
    for joint_name, joint_idx in angle_indices.items():
        sagittal = angle_data[0, joint_idx, :n_frames]
        frontal = angle_data[1, joint_idx, :n_frames]
        transverse = angle_data[2, joint_idx, :n_frames]
        all_angles[joint_name] = {
            'sagittal': sagittal,
            'frontal': frontal,
            'transverse': transverse
        }
    return all_angles

print("Available POINT labels:", labels)
# Extract 3D angles for cmj_1 only if angle labels exist
if len(labels) > 0 and any('Angles' in lbl for lbl in labels):
    all_angles_3d = extract_3d_angles(angle_data, labels)
else:
    print("No angle channels found under POINT.LABELS; skipping angle extraction.")
    all_angles_3d = {}




def extract_matrices(rotation_data, labels_rotation):
    # Build dict of joint_name -> (n_frames, 4, 4) for cmj_1 only
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
        positions = joint_matrices[:, :3, 3]
        positions_dict[joint_name] = positions
    return positions_dict


def _find_label_key(candidates, available_keys):
    # Return first available key matching any candidate (case-insensitive)
    low = {k.lower(): k for k in available_keys}
    for c in candidates:
        k = c.lower()
        if k in low:
            return low[k]
    # fuzzy: pick key containing all tokens
    tokens = candidates[0].lower().split('_') if candidates else []
    for k in available_keys:
        lk = k.lower()
        if all(t in lk for t in tokens):
            return k
    return None

def find_initial_foot_contact(matrices_dict, standing_frames=10, threshold=5):
    positions = extract_positions_from_matrices(matrices_dict)
    # Using right toe to determine contact; resolve best-matching key
    toe_key = _find_label_key(['r_toe', 'right_toe', 'righttoe', 'rtoe'], positions.keys())
    if toe_key is None:
        raise KeyError(f"Could not find a right toe key in positions. Available: {list(positions.keys())}")
    right_toe_positions = positions[toe_key]  # (n_frames, 3)
    z_coordinates = right_toe_positions[:, 2]
    # Calculate standing level as mean of first N frames
    standing_z = np.mean(z_coordinates[:standing_frames])
    # Find peak (highest point, i.e., flight apex)
    peak_frame = np.argmax(z_coordinates)
    # After peak, look for first frame where toe returns to standing_z (within threshold)
    after_peak = np.arange(peak_frame, len(z_coordinates))
    back_to_standing = after_peak[np.where(np.abs(z_coordinates[peak_frame:] - standing_z) < threshold)[0]]
    if len(back_to_standing) == 0:
        # Never returns, fallback to last frame
        return int(len(z_coordinates) - 1)
    initial_contact_frame = int(back_to_standing[0])
    return initial_contact_frame

initial_contact_frame = find_initial_foot_contact(extract_matrices(rotation_data, labels_rotation))
print(f"Initial foot contact frame: {initial_contact_frame}, Time: {initial_contact_frame/fs:.2f}s")

#### Plotting the knee angles at initial contact for each trial
####Knee valgus/varus('-'move outwards) should be within -5 to 10 degre

# Print angles at initial contact only if angles were extracted
if all_angles_3d:
    left_knee_flexion = -all_angles_3d['left_knee']['sagittal'][initial_contact_frame]
    left_knee_valgus = all_angles_3d['left_knee']['frontal'][initial_contact_frame]
    left_knee_rotation = all_angles_3d['left_knee']['transverse'][initial_contact_frame]
    left_foot_progression = all_angles_3d['left_fp']['transverse'][initial_contact_frame]
    right_knee_flexion = all_angles_3d['right_knee']['sagittal'][initial_contact_frame]
    right_knee_valgus = all_angles_3d['right_knee']['frontal'][initial_contact_frame]
    right_knee_rotation = all_angles_3d['right_knee']['transverse'][initial_contact_frame]
    right_foot_progression = all_angles_3d['right_fp']['transverse'][initial_contact_frame]
    left_ankle_flexion = all_angles_3d['left_ankle']['sagittal'][initial_contact_frame]
    right_ankle_flexion = all_angles_3d['right_ankle']['sagittal'][initial_contact_frame]

    print("Knee angles at initial contact (cmj_1):")
    print(f"Left Knee Flexion = {left_knee_flexion}, Right Knee Flexion = {right_knee_flexion}")
    print(f"Left Knee Valgus = {left_knee_valgus}, Right Knee Valgus = {right_knee_valgus}")
    print(f"Left Knee Rotation = {left_knee_rotation}, Right Knee Rotation = {right_knee_rotation}")
    print(f"Left Foot Progression = {left_foot_progression}, Right Foot Progression = {right_foot_progression}")
    print(f"Left Ankle Flexion = {left_ankle_flexion}, Right Ankle Flexion = {right_ankle_flexion}")
else:
    print("Angle channels unavailable; only position-based metrics computed.")

    # Compute average hip flexion over 10 frames centered at initial contact (landing) frame
window = 10
half_window = window // 2
start = max(0, initial_contact_frame - half_window)
end = min(len(all_angles_3d['left_hip']['sagittal']), initial_contact_frame + half_window + 1)

left_hip_flexion_avg = np.mean(all_angles_3d['left_hip']['sagittal'][start:end])
right_hip_flexion_avg = np.mean(all_angles_3d['right_hip']['sagittal'][start:end])
print(f"Left Hip Flexion at initial contact = {left_hip_flexion_avg}")
print(f"Right Hip Flexion at initial contact = {right_hip_flexion_avg}")
print("Initial contact frame:", initial_contact_frame)


        # Peak knee flexion AFTER initial contact
# Plot knee flexion over time after initial contact
import matplotlib.pyplot as plt

left_knee_flexion_trace = all_angles_3d['left_knee']['sagittal']
# Find peaks (local maxima) in the left hip flexion trace
left_hip_flexion_trace = all_angles_3d['left_hip']['sagittal']
peaks, _ = find_peaks(left_hip_flexion_trace)

plt.figure(figsize=(10, 5))
plt.plot(time, left_hip_flexion_trace, label='Left Hip Flexion')

plt.xlabel('Time (s)')
plt.ylabel('Hip Flexion (degrees)')
plt.title('Left Hip Flexion Over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print peak values and their frame/time
print("Left Hip Flexion Peaks:")
for idx in peaks:
    print(f"  Frame {idx}, Time {time[idx]:.2f}s: {left_hip_flexion_trace[idx]:.2f} deg")



#     positions = extract_positions_from_matrices(matrices_dict)
#     if foot_label not in positions:
#         print(f"Warning: {foot_label} not found in cmj_1")
#         return None
#     z = positions[foot_label][:, 2]
#     standing_z = np.mean(z[:standing_frames])
#     max_z = np.max(z)
#     jump_height = max_z - standing_z
#     print(f"cmj_1: Standing Z = {standing_z:.2f}, Max Z = {max_z:.2f}, Jump Height = {jump_height:.2f}, mm")
#     return jump_height

# # Example usage (single trial)
# jump_height = calculate_jump_height_from_foot(all_matrices, foot_label='r_foot', standing_frames=10)


# '''
# import matplotlib.pyplot as plt
# def plot_annotated_foot_height(all_matrices, trial_name, foot_label='r_foot', standing_frames=10):
#     all_positions = extract_positions_from_matrices(all_matrices)
#     z = all_positions[trial_name][foot_label][:, 2]
#     frames = np.arange(len(z))
#     standing_z = np.mean(z[:standing_frames])
#     max_z = np.max(z)
#     max_frame = np.argmax(z)
#     plt.figure(figsize=(10,5))
#     plt.plot(frames, z, label=f'{foot_label} Z-position')
#     plt.axhline(standing_z, color='green', linestyle='--', label='Standing Z')
#     plt.plot(max_frame, max_z, 'ro', label='Max Z')
#     plt.xlabel('Frame')
#     plt.ylabel('Vertical Position (mm)')
#     plt.title(f'Foot Height Over Time - {trial_name}')
#     plt.legend()
#     plt.grid(True)
#     plt.show()
# # Example usage:
# #plot_annotated_foot_height(all_matrices, 'trial_1', foot_label='r_foot')
# #plot_annotated_foot_height(all_matrices, 'trial_2', foot_label='r_foot')
# #plot_annotated_foot_height(all_matrices, 'trial_3', foot_label='r_foot')
# '''

# import numpy as np
# import matplotlib.pyplot as plt

# import numpy as np
# import matplotlib.pyplot as plt

# import numpy as np
# import matplotlib.pyplot as plt

# def detect_cmj_phases_height_based(all_positions, trial_name, pelvis_marker='pelvis', foot_marker='r_foot', threshold=10):
#     pelvis_z = all_positions[trial_name][pelvis_marker][:, 2]
#     foot_z = all_positions[trial_name][foot_marker][:, 2]
#     n_frames = len(pelvis_z)
#     standing_z = np.mean(pelvis_z[:10])
#     movement_start = np.where(np.abs(pelvis_z - standing_z) > threshold)[0]
#     standing_end = movement_start[0] if len(movement_start) > 0 else 0
#     min_pelvis_z_frame = np.argmin(pelvis_z[standing_end:]) + standing_end
#     max_pelvis_z_frame = np.argmax(pelvis_z)  # PEAK HEIGHT
#     standing_foot_z = np.mean(foot_z[:10])
#     takeoff_candidates = np.where(foot_z[min_pelvis_z_frame:] > standing_foot_z + threshold)[0]
#     propulsion_end = takeoff_candidates[0] + min_pelvis_z_frame if len(takeoff_candidates) > 0 else n_frames - 1
#     landing_candidates = np.where(foot_z[propulsion_end:] < standing_foot_z + threshold)[0]
#     landing_start = landing_candidates[0] + propulsion_end if len(landing_candidates) > 0 else n_frames - 1
#     settle_candidates = np.where(np.abs(foot_z[landing_start:] - standing_foot_z) < threshold/2)[0]
#     landing_end = settle_candidates[0] + landing_start if len(settle_candidates) > 0 else n_frames - 1
#     phases = {
#         'standing_end': standing_end,
#         'descent_end': min_pelvis_z_frame,
#         'peak_height': max_pelvis_z_frame,
#         'propulsion_end': propulsion_end,
#         'landing_start': landing_start,
#         'landing_end': landing_end
#     }
#     return phases

# def get_phase_intervals(phases, n_frames):
#     return [
#         ("Standing", 0, phases['standing_end']),
#         ("Descent", phases['standing_end'], phases['descent_end']),
#         ("Propulsion", phases['descent_end'], phases['propulsion_end']),
#         ("Flight", phases['propulsion_end'], phases['landing_start']),
#         ("Landing", phases['landing_start'], phases['landing_end']),
#         ("Settle", phases['landing_end'], n_frames-1)
#     ]

# def plot_aligned_cmj_trials_on_peak(all_positions, trial_names, pelvis_marker='pelvis', foot_marker='r_foot', threshold=10, window=100):
#     plt.figure(figsize=(16, 7))
#     colors = plt.cm.tab10.colors
#     for idx, trial in enumerate(trial_names):
#         phases = detect_cmj_phases_height_based(all_positions, trial, pelvis_marker, foot_marker, threshold)
#         pelvis_z = all_positions[trial][pelvis_marker][:, 2]
#         foot_z = all_positions[trial][foot_marker][:, 2]
#         n_frames = len(pelvis_z)
#         peak_frame = phases['peak_height']
#         # Extract window around peak
#         start = max(0, peak_frame - window)
#         end = min(n_frames, peak_frame + window)
#         x_vals = np.arange(start - peak_frame, end - peak_frame)
#         plt.plot(x_vals, pelvis_z[start:end], label=f"{trial} pelvis", color=colors[idx % len(colors)], linestyle='-')
#         plt.plot(x_vals, foot_z[start:end], label=f"{trial} foot", color=colors[idx % len(colors)], linestyle='--')

#         # Shade and label phases for this trial
#         phase_intervals = get_phase_intervals(phases, n_frames)
#         phase_colors = ["#f0f0f0", "#ffe0e0", "#e0ffe0", "#d0f5ff", "#ffe0b2", "#f9f9f9"]
#         for i, (phase, p_start, p_end) in enumerate(phase_intervals):
#             # Only shade if within visible window
#             if p_end >= start and p_start <= end:
#                 x1 = max(p_start - peak_frame, x_vals[0])
#                 x2 = min(p_end - peak_frame, x_vals[-1])
#                 plt.axvspan(x1, x2, color=phase_colors[i], alpha=0.13, lw=0)
            

#         # Draw vertical alignment line at peak


#     plt.xlabel('Frames (relative to peak pelvis height)')
#     plt.ylabel('Vertical Position (mm)')
#     plt.title('CMJ Trials: Aligned on Peak Pelvis Height (with phase regions)')
#     plt.legend(ncol=2)
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

# # Example usage (multi-trial plot skipped in single-trial workflow)
# # positions_cmj1 = extract_positions_from_matrices(all_matrices)
# # plot an adapted single-trial view if needed



# def print_cmj_phase_durations_single(positions_cmj1, pelvis_marker='pelvis', foot_marker='r_foot', threshold=10):
#     # Compute and print durations for cmj_1 only
#     all_positions = {'cmj_1': positions_cmj1}
#     phases = detect_cmj_phases_height_based(all_positions, 'cmj_1', pelvis_marker, foot_marker, threshold)
#     n_frames = len(positions_cmj1[pelvis_marker])
#     intervals = get_phase_intervals(phases, n_frames)
#     print("cmj_1 phase intervals (frames):")
#     for name, start, end in intervals:
#         print(f"  {name}: {start} -> {end} (len {end-start})")

# # Example usage (single-trial):
# positions_cmj1 = extract_positions_from_matrices(all_matrices)
# print_cmj_phase_durations_single(positions_cmj1, pelvis_marker='pelvis', foot_marker='r_foot', threshold=10)
