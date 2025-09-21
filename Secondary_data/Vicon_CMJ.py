import ezc3d
import numpy as np
import re
import matplotlib.pyplot as plt

# Update these paths if needed
cmj_1 = ezc3d.c3d("/Users/adamdray/Downloads/Baseline-1/Tash_CMJ1.c3d")
cmj_2 = ezc3d.c3d("/Users/adamdray/Downloads/Baseline-1/Tash_CMJ2.c3d")
cmj_3 = ezc3d.c3d("/Users/adamdray/Downloads/Baseline-1/Tash_CMJ3.c3d")

print(cmj_1['data'].keys())

points = cmj_1['data']['points']          # shape: (4, n_points, n_frames)
labels = cmj_1['parameters']['POINT']['LABELS']['value']
fps  = cmj_1['data']['analogs']              # dict with 'force','moment','cop'
fps_labels = cmj_1['parameters']['ANALOG']['LABELS']['value']
chan_matrix = cmj_1['parameters']['ANALOG']['USED']['value']
point_rate = float(cmj_1['parameters']['POINT']['RATE']['value'][0])
print("Rate check:", float(cmj_1['parameters']['ANALOG']['RATE']['value'][0]))
n_frames = points.shape[2]
time = np.arange(n_frames) / point_rate

n_sub, n_chn, n_fr = fps.shape
fs_an = float(cmj_1['header']['analogs']['frame_rate'])

A = fps.transpose(1, 0, 2).reshape(n_chn, n_sub*n_fr)
t_an = np.arange(A.shape[1]) / fs_an

fps_params = cmj_1['parameters'].get('FORCE_PLATFORM', {})
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


# === JUMP HEIGHT FROM MARKER DATA ===

pelvis_i = labels.index('LPSI')
z = points[2, pelvis_i, :]
standing_z = np.mean(z[:200])
max_z = np.max(z)
jump_height = max_z - standing_z

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

print("\n=== Take-Off & Landing Detection ===")
print(f"\tTake-off index: {idx_tako}, time: {t_an[idx_tako]:.3f} s")
print(f"\tLanding index: {idx_land}, time: {t_an[idx_land]:.3f} s")


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


# Total force plot
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

