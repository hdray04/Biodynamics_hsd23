import ezc3d
import numpy as np
import re
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# Update these paths if needed
single_hop_left_1 = ezc3d.c3d("/Users/harrietdray/Baseline/Tash_single_hop_left_1.c3d")
single_hop_left_2 = ezc3d.c3d("/Users/harrietdray/Baseline/Tash_single_hop_left_2.c3d")
single_hop_left_3 = ezc3d.c3d("/Users/harrietdray/Baseline/Tash_single_hop_left_3.c3d")
single_hop_right_1 = ezc3d.c3d("/Users/harrietdray/Baseline/Tash_single_hop_right_1.c3d")
single_hop_right_2 = ezc3d.c3d("/Users/harrietdray/Baseline/Tash_single_hop_right_2.c3d")
single_hop_right_3 = ezc3d.c3d("/Users/harrietdray/Baseline/Tash_single_hop_right_3.c3d")
#print(single_hop__left_2['data'].keys())

points = single_hop_right_3['data']['points']          # shape: (4, n_points, n_frames)
labels = single_hop_right_3['parameters']['POINT']['LABELS']['value']
labels_rotation = single_hop_right_3['parameters']['POINT']['ANGLES']['value']
point_rate = float(single_hop_right_3['parameters']['POINT']['RATE']['value'][0])
print("Rate check:", float(single_hop_right_3['parameters']['ANALOG']['RATE']['value'][0]))
n = np.arange(points.shape[2])
n_frames = points.shape[2]
time = np.arange(n_frames) / point_rate


# === JUMP HEIGHT FROM MARKER DATA ===  

pelvis_i = labels.index('PELVISO')
z = points[2, pelvis_i, :]
standing_z = np.mean(z[:200])
max_z = np.max(z)
jump_height = max_z - standing_z

# # Plot PELVISO height over time
# plt.figure(figsize=(10, 4))
# plt.plot(n, z, label='PELVISO Z')
# plt.xlabel('Frame')
# plt.ylabel('Height (mm)')
# plt.title('PELVISO Height Over Time')
# plt.legend()
# plt.tight_layout()
# plt.show()

# === HEEL MARKER X MOVEMENT AND JUMP DISTANCE ===

# Choose heel marker (e.g., 'LHEELO' or 'RHEELO')
heel_label = 'RHEE'
heel_i = labels.index(heel_label)
heel_x = points[0, heel_i, :]

# Plot heel X position over time
plt.figure(figsize=(10, 4))
plt.plot(n, heel_x, label=f'{heel_label} X')
plt.xlabel('Frame')
plt.ylabel('X Position (mm)')
plt.title(f'{heel_label} X Position Over Time')
plt.legend()
plt.tight_layout()
plt.show()

# Calculate jump distance using median X before and after jump
# Assume standing before jump is first 100 frames
standing_x_before = np.median(heel_x[:100])

# Find landing frame: when heel Z returns to within 5 mm of its mean standing value after takeoff
heel_z = points[2, heel_i, :]
standing_z_heel = np.mean(heel_z[:100])
# Find takeoff: first frame where heel_z rises >10 mm above standing
takeoff_frame = np.argmax(heel_z > standing_z_heel + 10)
# Find landing: first frame after takeoff where heel_z returns within 5 mm of standing
landing_candidates = np.where((np.abs(heel_z - standing_z_heel) < 5) & (np.arange(len(heel_z)) > takeoff_frame))[0]
Landing_frame = landing_candidates[0] if len(landing_candidates) > 0 else None

if Landing_frame is not None and Landing_frame + 100 < len(heel_x):
    standing_x_after = np.median(heel_x[Landing_frame:Landing_frame+100])
else:
    standing_x_after = float('nan')

jump_distance = standing_x_after - standing_x_before
print(f"Standing {heel_label} X before jump (median of first 100 frames): {standing_x_before:.2f} mm")
print(f"Standing {heel_label} X after jump (median of 100 frames after landing): {standing_x_after:.2f} mm")
print(f"Jump distance (median X after - before): {jump_distance:.2f} mm")

## === ANGLES AT LANDING ===
print("Landing frame:", Landing_frame)
angles_deg = {}
if Landing_frame is not None:
    angles_deg = {name:points[:,labels.index(name),Landing_frame] for name in labels_rotation}

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
