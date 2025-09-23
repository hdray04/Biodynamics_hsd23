import ezc3d
import numpy as np
import re
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# Update these paths if needed
sldj_left_1 = ezc3d.c3d("/Users/harrietdray/Library/CloudStorage/OneDrive-ImperialCollegeLondon/ACL_data/Pilot - Tash/Baseline/Tash_SLDJ2_left.c3d")
sldj_left_3 = ezc3d.c3d('/Users/harrietdray/Library/CloudStorage/OneDrive-ImperialCollegeLondon/ACL_data/Pilot - Tash/Baseline/Tash_SLDJ3_left.c3d')
sldj_right_1 = ezc3d.c3d('/Users/harrietdray/Library/CloudStorage/OneDrive-ImperialCollegeLondon/ACL_data/Pilot - Tash/Baseline/Tash_SLDJ1_right.c3d')

print(sldj_right_1['data'].keys())

points = sldj_right_1['data']['points']          # shape: (4, n_points, n_frames)
labels = sldj_right_1['parameters']['POINT']['LABELS']['value']
labels_rotation = sldj_right_1['parameters']['POINT']['ANGLES']['value']
point_rate = float(sldj_right_1['parameters']['POINT']['RATE']['value'][0])
print("Rate check:", float(sldj_right_1['parameters']['ANALOG']['RATE']['value'][0]))
n = np.arange(points.shape[2])
n_frames = points.shape[2]
time = np.arange(n_frames) / point_rate


# === JUMP HEIGHT FROM MARKER DATA ===

pelvis_i = labels.index('PELVISO')
z = points[2, pelvis_i, :]
standing_z = np.mean(z[:200])
max_z = np.max(z)
jump_height = max_z - standing_z

# Plot PELVISO height over time
plt.figure(figsize=(10, 4))
plt.plot(n, z, label='PELVISO Z')
plt.xlabel('Frame')
plt.ylabel('Height (mm)')
plt.title('PELVISO Height Over Time')
plt.legend()
plt.tight_layout()
plt.show()


# Choose frame window, e.g. from 200 to 600
start_frame = 350
end_frame = 480

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


## === ANGLES AT LANDING ===
Landing_frame = None
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
