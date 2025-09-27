import ezc3d
import numpy as np
import re
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# Update these paths if needed
triple_hop_left_1 = ezc3d.c3d("/Users/harrietdray/Baseline/Tash_triple_hop_left_1.c3d")
triple_hop_left_2 = ezc3d.c3d("/Users/harrietdray/Baseline/Tash_triple_hop_left_2.c3d")
triple_hop_left_3 = ezc3d.c3d("/Users/harrietdray/Baseline/Tash_triple_hop_left_3.c3d")
triple_hop_right_1 = ezc3d.c3d("/Users/harrietdray/Baseline/Tash_triple_hop_right_1.c3d")
triple_hop_right_2 = ezc3d.c3d("/Users/harrietdray/Baseline/Tash_triple_hop_right_2.c3d")
triple_hop_right_3 = ezc3d.c3d("/Users/harrietdray/Baseline/Tash_triple_hop_right_3.c3d")
#print(single_hop__left_2['data'].keys())

points = triple_hop_left_3['data']['points']          # shape: (4, n_points, n_frames)
labels = triple_hop_left_3['parameters']['POINT']['LABELS']['value']
labels_rotation = triple_hop_left_3['parameters']['POINT']['ANGLES']['value']
point_rate = float(triple_hop_left_3['parameters']['POINT']['RATE']['value'][0])
print("Rate check:", float(triple_hop_left_3['parameters']['ANALOG']['RATE']['value'][0]))
n = np.arange(points.shape[2])
n_frames = points.shape[2]
time = np.arange(n_frames) / point_rate


# === JUMP HEIGHT FROM MARKER DATA ===  

pelvis_i = labels.index('LHEE')
z = points[2, pelvis_i, :]  
standing_z = np.mean(z[:200])
max_z = np.max(z)
jump_height = max_z - standing_z

# # Plot PELVISO height over time
plt.figure(figsize=(10, 4))
plt.plot(n, z, label='PELVISO Z')
plt.xlabel('Frame')
plt.ylabel('Height (mm)')
plt.title('PELVISO Height Over Time')
plt.legend()
plt.tight_layout()
plt.show()

 
peaks, _ = find_peaks(z, distance=50, prominence=20)

# extract peak heights
peak_heights = z[peaks]

plt.plot(n, z, label='PELVISO Z')
plt.plot(n[peaks], peak_heights, "x", label='Detected Peaks')
plt.xlabel('Frame')
plt.ylabel('Height (mm)')
plt.title('PELVISO Height with Detected Peaks')

print("Peak frames:", n[peaks])
print("Peak heights:", peak_heights)

for i, peak in enumerate(peaks):
    jump_height_i = z[peak] - standing_z
    print(f"Jump {i+1}: Peak frame {peak}, Height: {jump_height_i:.2f} mm")
# === HEEL MARKER X MOVEMENT AND JUMP DISTANCE ===

# Choose heel marker (e.g., 'LHEE' or 'RHEE')
heel_label = 'LHEE'
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

# Calculate distance of each jump using X value of RHEE
heel_z = points[2, heel_i, :]
heel_x = points[0, heel_i, :]

# Standing Z value (mean of first 100 frames)
standing_z_heel = np.mean(heel_z[:100])

# Find all frames where heel_z is within 5 mm of standing (potential landings)
close_to_standing = np.abs(heel_z - standing_z_heel) < 9

# Find transitions: rising edge = takeoff, falling edge = landing
# We'll use the difference to find where close_to_standing changes from False to True (landing)
landing_frames = np.where((~close_to_standing[:-1]) & (close_to_standing[1:]))[0] + 1

# To avoid false positives at the start, ignore landings in the first 50 frames
landing_frames = landing_frames[landing_frames > 50]

# Calculate jump distances including the first jump (from standing to first landing)
jump_distances = []

# First jump: from standing (mean of first 10 frames) to first landing
if len(landing_frames) > 0:
    standing_x = np.median(heel_x[:10])
    first_landing_x = np.median(heel_x[landing_frames[0]:landing_frames[0]+10])
    first_distance = first_landing_x - standing_x
    jump_distances.append(first_distance)
    print(f"Jump 1: from standing to frame {landing_frames[0]}, distance: {first_distance:.2f} mm")

# Subsequent jumps: between consecutive landings
for i in range(1, len(landing_frames)):
    before = np.median(heel_x[landing_frames[i-1]:landing_frames[i-1]+10])
    after = np.median(heel_x[landing_frames[i]:landing_frames[i]+10])
    distance = after - before
    jump_distances.append(distance)
    print(f"Jump {i+1}: from frame {landing_frames[i-1]} to {landing_frames[i]}, distance: {distance:.2f} mm")



# For plotting: mark landings on the heel_x plot
plt.figure(figsize=(10, 4))
plt.plot(n, heel_x, label=f'{heel_label} X')
plt.plot(landing_frames, heel_x[landing_frames], 'rx', label='Landings')
plt.xlabel('Frame')
plt.ylabel('X Position (mm)')
plt.title(f'{heel_label} X Position with Landings')
plt.legend()
plt.tight_layout()
plt.show()

# Set Landing_frame as the first detected landing for downstream code
Landing_frame = landing_frames[0] if len(landing_frames) > 0 else None

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


            # === Plot Knee Flexion (X axis) Over Frames and Extract Peaks ===

# Plot and analyze left knee flexion only
knee = 'LKneeAngles'
knee_flexion = {}

if knee in labels_rotation:
    idx = labels.index(knee)
    flexion = points[0, idx, :]  # X axis = flexion/extension
    knee_flexion[knee] = flexion
    plt.plot(n, flexion, label=f'{knee} Flexion (X)')
    plt.xlabel('Frame')
    plt.ylabel('Flexion Angle (deg)')
    plt.title('Right Knee Flexion Over Time')
    plt.legend()
    plt.tight_layout()
    plt.show()
    # Plot landing frames (ground contact points) on the knee flexion graph
    if len(landing_frames) > 0:
        plt.plot(n, flexion, label=f'{knee} Flexion (X)')
        plt.plot(landing_frames, flexion[landing_frames], 'go', label='Landings (Ground Contact)')
        plt.xlabel('Frame')
        plt.ylabel('Flexion Angle (deg)')
        plt.title(f'{knee} Flexion with Landings Marked')
        plt.legend()
        plt.tight_layout()
        plt.show()
    # Extract peaks for right knee and print them
    peaks, _ = find_peaks(flexion, distance=20, prominence=5)
    print(f"\n{knee} flexion peaks:")
    for i, peak in enumerate(peaks):
        print(f"  Peak {i+1}: Frame {peak}, Angle {flexion[peak]:.2f}°")
    # Optional: plot peaks
    plt.plot(n, flexion, label=f'{knee} Flexion (X)')
    plt.plot(n[peaks], flexion[peaks], "rx", label='Peaks')
    plt.xlabel('Frame')
    plt.ylabel('Flexion Angle (deg)')
    plt.title(f'{knee} Flexion with Detected Peaks')
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    knee_flexion[knee] = None
    print(f"{knee} not found in rotation labels")

