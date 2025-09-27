import ezc3d
import numpy as np
import re
import matplotlib.pyplot as plt

# Update these paths if needed
cmj_1 = ezc3d.c3d("/Users/harrietdray/Baseline/Tash_CMJ1.c3d")
cmj_2 = ezc3d.c3d("/Users/harrietdray/Baseline/Tash_CMJ2.c3d")
cmj_3 = ezc3d.c3d("/Users/harrietdray/Baseline/Tash_CMJ3.c3d")
slvj_left_1 = ezc3d.c3d("/Users/harrietdray/Baseline/Tash_SLVJ1_left.c3d")
slvj_right_1 = ezc3d.c3d("/Users/harrietdray/Baseline/Tash_SLVJ1_right.c3d")

print(cmj_2['data'].keys())

points = cmj_3['data']['points']          # shape: (4, n_points, n_frames)
labels = cmj_3['parameters']['POINT']['LABELS']['value']
labels_rotation = cmj_3['parameters']['POINT']['ANGLES']['value']
fps  = cmj_3['data']['analogs']              # dict with 'force','moment','cop'
fps_labels = cmj_3['parameters']['ANALOG']['LABELS']['value']
chan_matrix = cmj_3['parameters']['ANALOG']['USED']['value']
point_rate = float(cmj_3['parameters']['POINT']['RATE']['value'][0])
print("Rate check:", float(cmj_3['parameters']['ANALOG']['RATE']['value'][0]))
n_frames = points.shape[2]
time = np.arange(n_frames) / point_rate

n_sub, n_chn, n_fr = fps.shape
fs_an = float(cmj_3['header']['analogs']['frame_rate'])

A = fps.transpose(1, 0, 2).reshape(n_chn, n_sub*n_fr)
t_an = np.arange(A.shape[1]) / fs_an

fps_params = cmj_3['parameters'].get('FORCE_PLATFORM', {})
ch_values = np.array(fps_params['CHANNEL']['value'])
if ch_values.ndim == 1:
    if ch_values.size % 6 != 0:
        raise RuntimeError("Unexpected FORCE_PLATFORM:CHANNEL length")
    ch_values = ch_values.reshape(6, -1, order='F').T
elif ch_values.shape[0] == 6:
    ch_values = ch_values.T
elif ch_values.shape[1] == 6:
    pass
else:
    raise RuntimeError("Unrecognized FORCE_PLATFORM:CHANNEL dimensions")

forceplates = []
for pidx, row in enumerate(ch_values):
    iFx, iFy, iFz, iMx, iMy, iMz = (int(i)-1 for i in row)

    F = np.zeros((3, A.shape[1]))
    if iFx is not None: F[0, :] = A[iFx, :]
    if iFy is not None: F[1, :] = A[iFy, :]
    if iFz is not None: F[2, :] = A[iFz, :]

    forceplates.append({'F': F, 'channels':{'Fx': iFx, 'Fy': iFy, 'Fz': iFz,
                                           'Mx': iMx, 'My': iMy, 'Mz': iMz}, 'plate': pidx})


print(f"Number of force plates: {len(forceplates)}")


# === KEY VARIABLES ===

fs = fs_an
dt = 1.0 / fs
g = 9.81


## === CALC FORCES ===

Fz_total = - np.sum([fp['F'][2, :] for fp in forceplates], axis=0) # sum across plates
quiet_n = int(min(len(Fz_total), 0.5 * fs_an)) # find body weight from quiet window
BW = float(np.median(Fz_total[:quiet_n]))
Fz_total_norm = Fz_total - BW # normalize to body weight
print("Normalized Fz_total:", Fz_total_norm)
mass = BW / g
print(f"Body weight: {BW:.2f} N, Mass: {mass:.2f} kg")
# Assign force plate indices
right_fp = forceplates[0]  # Right foot
left_fp = forceplates[1]   # Left foot

# Calculate LSI (Limb Symmetry Index) for Peak Fz
peak_fz_right = np.max(-right_fp['F'][2, :]) / mass
peak_fz_left = np.max(-left_fp['F'][2, :]) / mass

LSI_of_Peak_Fz = (peak_fz_left / peak_fz_right) * 100  # Expressed as percentage

print(f"Peak Fz Right: {peak_fz_right:.3f} N/kg")
print(f"Peak Fz Left: {peak_fz_left:.3f} N/kg")
print(f"Limb Symmetry Index (LSI) of Peak Fz: {LSI_of_Peak_Fz:.2f}%")

# === JUMP HEIGHT FROM MARKER DATA ===

pelvis_i = labels.index('PELVISO')
z = points[2, pelvis_i, :]
standing_z = np.mean(z[:200])
max_z = np.max(z)
jump_height = max_z - standing_z

# Plot pelvis height (Z) over time
plt.figure(figsize=(10, 4))
plt.plot(time, z, label='Pelvis Height (Z)')
plt.axhline(standing_z, color='g', linestyle='--', label='Standing Z')
plt.axhline(max_z, color='r', linestyle='--', label='Max Z')
plt.xlabel('Time (s)')
plt.ylabel('Pelvis Height (mm)')
plt.title('Pelvis Height Over Time')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n=== Marker Calculation ===")
print(f"\tStanding Z: {standing_z:.2f} mm")
print(f"\tMax Z: {max_z:.2f} mm")
print(f"\n\tEstimated Jump Height: {jump_height:.2f} mm")


# === DETERMINE TAKE-OFF & LANDING INDEX ===

thresh_N = 0.05 * BW  # 5% body weight threshold
contact = Fz_total < thresh_N
stay = max(int(0.05*fs), 1) # minimum frames to confirm not noise
idx_tako= next((i for i in range(stay, len(Fz_total)-stay)
               if contact[i] and np.all(contact[i:i+stay])), None)
idx_land = next((i for i in range(idx_tako+1, len(contact)-stay) if ~contact[i] and np.all(~contact[i:i+stay])), None)

frame_tako = int(round(idx_tako / (fs_an / point_rate)))
frame_land = int(round(idx_land / (fs_an / point_rate)))

print("\n=== Take-Off & Landing Detection ===")
print(f"\tTake-off index: {idx_tako}, Take-off frame: {frame_tako}, time: {t_an[idx_tako]:.3f} s")
print(f"\tLanding index: {idx_land}, Landing frame: {frame_land}, time: {t_an[idx_land]:.3f} s")


# === JUMP HEIGHT FROM FLIGHT TIME ===

t_flight = t_an[idx_land] - t_an[idx_tako]
h_flight = (g * t_flight**2) / 8

print("\n=== Flight Time Calculation ===")
print(f"\tFlight time: {t_flight:.3f} s")
print(f"\n\tEstimated Jump Height: {h_flight*1000:.2f} mm")


# === JUMP HEIGHT FROM IMPULSE-MOMENTUM ===

def find_negative_drop(x, fs, window_ms=50, frac=0.9, eps=0.0, smooth_ms=1, end=None):
    """
    x: 1D array
    fs: Hz
    window_ms: duration to stay negative
    frac: fraction of samples in window that must be negative
    eps: negativity threshold
    smooth_ms: moving average window in ms
    end: only search x[:end]
    """
    if end is None:
        end = len(x)

    # minimal smoothing
    w_smooth = max(1, int(round(smooth_ms * fs / 1_000)))
    if w_smooth > 1:
        kernel = np.ones(w_smooth) / w_smooth
        x = np.convolve(x, kernel, mode="same")

    w = max(1, int(round(window_ms * fs / 1_000)))
    neg = (x[:end] < -eps).astype(np.int32)
    s = np.convolve(neg, np.ones(w, dtype=int), mode='valid')
    idxs = np.where(s >= frac * w)[0]
    return int(idxs[0]) if idxs.size else None

idx_v0 = find_negative_drop(Fz_total_norm, idx_tako)
print(idx_v0)
# idx_v0 = np.argmax(Fz_total_norm[100:idx_tako]) + 100 # start of concentric phase in contact phase

impulse = np.trapezoid(Fz_total_norm[idx_v0:idx_tako+1], dx=dt) 

mass = BW / g

velocity_tako = impulse / mass
height_m = (velocity_tako ** 2) / (2 * g)
height_mm = height_m * 1000  # convert to mm

print("\n=== Impulse Calculation ===")
print(f"\V0 Start Time: {t_an[idx_v0]:.3f} s")
print(f"\tVelocity at Take-Off: {velocity_tako:.2f} m/s")
print(f"\n\tEstimated Jump Height: {height_mm:.2f} mm")


# === PLOTTING ===


# # Total force plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(t_an, Fz_total, lw=1.5, label='Fz_total')
ax.axvline(t_an[idx_v0], color='r', linestyle='--', label='conc_start')
ax.axvline(t_an[idx_tako], color='g', linestyle='--', label='take-off')
ax.axhline(BW, color='b', linestyle=':', label='Body Weight')
ax.set_title("Total Vertical GRF (Fz_total)")
ax.set_ylabel("Fz (N)")
ax.set_xlabel("Time (s)")
ax.grid(True, alpha=0.3)
ax.legend()
axes = [ax]  # for compatibility with later code
fig.tight_layout()
plt.show()



## === ANGLES AT LANDING ===
Landing_frame = frame_land
print("Landing frame:", Landing_frame)
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


        # === Average Hip Flexion 10 Frames Around Landing ===
window = 10  # frames before and after landing
hip_flexion_avg = {}
for name in hip_names:
    if name in labels_rotation:
        idx = labels.index(name)
        # Convert landing index (analog) to marker frame
        landing_frame = int(round(idx_land / (fs_an / point_rate)))
        start = max(0, landing_frame - window)
        end = min(points.shape[2], landing_frame + window + 1)
        avg_flexion = np.mean(points[0, idx, start:end])
        hip_flexion_avg[name] = avg_flexion
        print(f"Average {name} flexion (X) from frame {start} to {end-1}: {avg_flexion:.2f}°")
    else:
        hip_flexion_avg[name] = float('nan')
        print(f"{name}: Not found in rotation labels")

        # Plot hip flexion (X angle) over frames for left and right hips
plt.figure(figsize=(10, 4))
for name in hip_names:
    if name in labels_rotation:
        idx = labels.index(name)
        plt.plot(np.arange(points.shape[2]), points[0, idx, :], label=f"{name} flexion (X)")
plt.axvline(Landing_frame, color='k', linestyle='--', label='Landing Frame')
plt.xlabel('Frame')
plt.ylabel('Hip Flexion (deg)')
plt.title('Hip Flexion (X) Over Frames')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
# Forces per plate
# idx_max_f = idx_conc
# idx_to = idx_tako
# n = len(forceplates)
# rows, cols = 2, 2
# fig, axs = plt.subplots(rows, cols, figsize=(12, 8))
# axes = axs.ravel()
# for k in range(rows*cols):
#     ax = axes[k]
#     if k < n:
#         Fz = forceplates[k]['F'][2, :]
#         Fz = -Fz
#         ax.plot(t_an, Fz, lw=1.0)
#         ax.axvline(t_an[idx_max_f], color='r', linestyle='--', label='conc_start')
#         ax.axvline(t_an[idx_to], color='g', linestyle='--', label='take-off')
#         ax.axhline(BW, color='b', linestyle=':', label='Body Weight')
#         ax.set_title(f"Force Plate {k+1}")
#         ax.set_ylabel("Fz (N)")
#         ax.grid(True, alpha=0.3)
#         ax.legend()
#     else:
#         ax.axis('off')
#         ax.axis('off')
# axes[-2].set_xlabel("Time (s)")
# axes[-1].set_xlabel("Time (s)")
# fig.suptitle("Force–Time (Vertical GRF) per Force Plate", y=0.98)
# fig.tight_layout()
# plt.show()

