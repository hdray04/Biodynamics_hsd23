import ezc3d
import numpy as np
import re
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# File paths
sldj_left_1 = ezc3d.c3d("/Users/harrietdray/Library/CloudStorage/OneDrive-ImperialCollegeLondon/ACL_data/Pilot - Tash/Baseline/Tash_SLDJ1_left.c3d")
sldj_left_2 = ezc3d.c3d('/Users/harrietdray/Library/CloudStorage/OneDrive-ImperialCollegeLondon/ACL_data/Pilot - Tash/Baseline/Tash_SLDJ2_left.c3d')
sldj_left_3 = ezc3d.c3d('/Users/harrietdray/Library/CloudStorage/OneDrive-ImperialCollegeLondon/ACL_data/Pilot - Tash/Baseline/Tash_SLDJ3_left.c3d')
sldj_right_1 = ezc3d.c3d('/Users/harrietdray/Library/CloudStorage/OneDrive-ImperialCollegeLondon/ACL_data/Pilot - Tash/Baseline/Tash_SLDJ1_right.c3d')
sldj_right_2 = ezc3d.c3d('/Users/harrietdray/Library/CloudStorage/OneDrive-ImperialCollegeLondon/ACL_data/Pilot - Tash/Baseline/Tash_SLDJ2_right.c3d')
sldj_right_3 = ezc3d.c3d('/Users/harrietdray/Library/CloudStorage/OneDrive-ImperialCollegeLondon/ACL_data/Pilot - Tash/Baseline/Tash_SLDJ3_right.c3d')
print(sldj_right_1['data'].keys())

#Load data 
points = sldj_right_1['data']['points']          # shape: (4, n_points, n_frames)
labels = sldj_right_1['parameters']['POINT']['LABELS']['value']
labels_rotation = sldj_right_1['parameters']['POINT']['ANGLES']['value']
point_rate = float(sldj_right_1['parameters']['POINT']['RATE']['value'][0])
print("Rate check (SLDJ right 1):", float(sldj_right_1['parameters']['ANALOG']['RATE']['value'][0]))
n = np.arange(points.shape[2])
n_frames = points.shape[2]
time = np.arange(n_frames) / point_rate




# Plot PELVISO height over time
plt.figure(figsize=(10, 4))
plt.plot(n, z, label='PELVISO Z')
#plt.plot(n_cmj, z_cmj, label='PELVISO Z (CMJ)')
plt.xlabel('Frame')
plt.ylabel('Height (mm)')
plt.title('PELVISO Height Over Time')
plt.legend()
plt.tight_layout()
plt.show()


# Choose frame window, e.g. from 200 to 600
start_frame = 300
end_frame = 450

# Extract Z trajectory in that window
z_window = z[start_frame:end_frame]

# Find minimum and maximum inside this window
min_z = np.min(z_window)
min_frame = np.argmin(z_window) + start_frame
max_z = np.max(z_window)
max_frame = np.argmax(z_window) + start_frame

jump_height = max_z - min_z

print(f"Min Z: {min_z:.2f} mm at frame {min_frame}")
print(f"Max Z: {max_z:.2f} mm at frame {max_frame}")
print(f"Jump height (max - min): {jump_height:.2f} mm")

# === GROUND CONTACT TIME ESTIMATION (Stable Toe Z) ===
# Use toe marker Z (e.g., 'LTOE' or 'RTOE') to estimate ground level

toe_label = 'LTOE' if 'LTOE' in labels else 'RTOE'
toe_i = labels.index(toe_label)
toe_z = points[2, toe_i, :]

peak_frame = 493

# Find stable ground level after landing: look for a window of at least 50 consecutive frames after peak
post_peak_z = toe_z[peak_frame + 50:]  # skip a few frames after peak for landing

stable_idx = None
for i in range(len(post_peak_z) - 49):
    window = post_peak_z[i:i+50]
    if np.max(window) - np.min(window) < 2:  # stability threshold (2mm range)
        stable_idx = i
        break

if stable_idx is not None:
    ground_level_z = np.mean(post_peak_z[stable_idx:stable_idx+50])
    ground_level_frame = peak_frame + 20 + stable_idx
    print(f"Stable ground level Z: {ground_level_z:.2f} mm at frame {ground_level_frame}")
else:
    ground_level_z = np.mean(post_peak_z[-50:]) if post_peak_z.size >= 50 else np.mean(post_peak_z)
    ground_level_frame = peak_frame + 20 + (len(post_peak_z) - 50 if post_peak_z.size >= 50 else 0)
    print(f"No stable 50-frame window found; using last 50 frames after landing. Ground level Z: {ground_level_z:.2f} mm at frame {ground_level_frame}")

# Calculate time toe spent within 5mm of ground level BEFORE jump but after drop
# Assume "after drop" is after the first time toe_z rises above ground_level_z + 5mm
pre_jump_frames = np.arange(0, peak_frame)
within_5mm = np.abs(toe_z[pre_jump_frames] - ground_level_z) <= 3

# Find the first frame after drop (toe leaves ground)
drop_frame_candidates = np.where(toe_z[pre_jump_frames] > ground_level_z + 3)[0]
if drop_frame_candidates.size > 0:
    drop_frame = drop_frame_candidates[0]
else:
    drop_frame = 0  # fallback: start from beginning

# Only count frames after drop_frame and before jump apex
contact_frames = np.where(within_5mm & (pre_jump_frames >= drop_frame))[0]
contact_time = len(contact_frames) / point_rate

print(f"Toe spent {contact_time:.3f} s within 5mm of ground level before jump (frames {contact_frames[0] if contact_frames.size > 0 else 'N/A'} to {contact_frames[-1] if contact_frames.size > 0 else 'N/A'})")

landing_frame = ground_level_frame 

# === JUMP HEIGHT FROM MARKER DATA ===
pelvis_i = labels.index('PELVISO')
z = points[2, pelvis_i, :]
max_z = np.max(z)
print("Max pelvis height (mm):", max_z)
jump_height = max_z - ground_level_z
print("Jump height (mm):", jump_height)

## === ANGLES AT LANDING ===
print("Landing frame:", landing_frame)
angles_deg = {}
if landing_frame is not None:
    angles_deg = {name:points[:,labels.index(name),landing_frame] for name in labels_rotation}

    # Knee valgus at landing: Y angle of knee markers at landing frame
    knee_valgus = {}
    for knee in ['LKneeAngles', 'RKneeAngles']:
        if knee in labels_rotation:
            idx = labels.index(knee)
            valgus_angle = points[1, idx, Landing_frame]  # Y axis
            knee_valgus[knee] = valgus_angle
            print(f"{knee} valgus (Y) at landing: {valgus_angle:.2f}°")
        else:
            knee_valgus[knee] = float('nan')
            print(f"{knee} not found in rotation labels")
    # Print hip angles at landing frame
    hip_names = ['LHipAngles', 'RHipAngles']
    print("\n=== Hip Angles at Landing Frame ===")
    for name in hip_names:
        if name in labels_rotation:
            idx = labels.index(name)
            angles = points[:, idx, Landing_frame]
            print(f"{name}: X={angles[0]:.2f}°, Y={angles[1]:.2f}°, Z={angles[2]:.2f}°")
        else:
            print(f"{name}: Not found in rotation labels")

    # Extract peak (maximum) X angle after landing frame for both knees without explicit loop
    joint_names = {
        'Knee': ['LKneeAngles', 'RKneeAngles'],
        'Hip': ['LHipAngles', 'RHipAngles'],
    }

    peak_flexion = {}
    for joint, names in joint_names.items():
        peak_flexion[joint] = {
            side: (
                {'max_x': np.max(points[0, labels.index(side), Landing_frame:]),
                 'frame': np.argmax(points[0, labels.index(side), Landing_frame:]) + Landing_frame}
                if side in labels_rotation else
                {'max_x': float('nan'), 'frame': None}
            )
            for side in names
        }

    print("\n=== Peak Flexion (X axis) After Landing ===")
    for joint, sides in peak_flexion.items():
        for side, res in sides.items():
            print(f"{joint} {side}: {res['max_x']:.2f} deg at frame {res['frame']}")


    # Print ankle angles at landing frame
    ankle_names = ['LAnkleAngles', 'RAnkleAngles']
    print("\n=== Ankle Angles at Landing Frame ===")
    for name in ankle_names:
        if name in labels_rotation:
            idx = labels.index(name)
            angles = points[:, idx, Landing_frame]
            print(f"{name}: X={angles[0]:.2f}°, Y={angles[1]:.2f}°, Z={angles[2]:.2f}°")
        else:
            print(f"{name}: Not found in rotation labels")
