import ezc3d
import numpy as np
import re

# Update these paths if needed
cmj_1 = ezc3d.c3d("/Users/harrietdray/Baseline/Tash_CMJ1.c3d")
cmj_2 = ezc3d.c3d("/Users/harrietdray/Baseline/Tash_CMJ2.c3d")
cmj_3 = ezc3d.c3d("/Users/harrietdray/Baseline/Tash_CMJ3.c3d")

print(cmj_1['data'].keys())

points = cmj_1['data']['points']          # shape: (4, n_points, n_frames)
labels = cmj_1['parameters']['POINT']['LABELS']['value']
fps  = cmj_1['data']['analogs']              # dict with 'force','moment','cop'
fps_labels = cmj_1['parameters']['ANALOG']['LABELS']['value']
chan_matrix = cmj_1['parameters']['ANALOG']['USED']['value']
point_rate = float(cmj_1['parameters']['POINT']['RATE']['value'][0])
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
# Sum Fz across all available plates
Fz_total = np.sum([fp['F'][2, :] for fp in forceplates], axis=0)

# Enforce upward positive if needed using a quiet window (first 0.5 s)
quiet_n = int(min(len(Fz_total), 0.5 * fs_an))
if quiet_n > 0 and np.median(Fz_total[:quiet_n]) < 0:
    Fz_total = -Fz_total


print(f"Total Fz: samples={Fz_total.size}, dt={1.0/fs_an:.6f} s")


pelvis_i = labels.index('LPSI')
z = points[2, pelvis_i, :]
standing_z = np.mean(z[:200])
max_z = np.max(z)
jump_height = max_z - standing_z
print(f"cmj_1: Standing Z = {standing_z:.2f}, Max Z = {max_z:.2f}, Jump Height = {jump_height:.2f}, mm")


g = 9.81
dt = float(np.mean(np.diff(t_an)))
quiet_n = int(0.5 * fs_an)
BW = float(np.median(Fz_total[:quiet_n]))
m = BW / g
print(f"Body weight = {BW:.1f} N")

thresh_N = 30
contact = Fz_total > thresh_N
edges = np.where((contact[:-1] == True) & (contact[1:] == False))[0]
if len(edges) == 0:
    raise RuntimeError("No take-off detected. Adjust threshold or quiet window.")
to_idx = int(edges[0])

Fnet = Fz_total - BW
a = Fnet[:to_idx+1] / m
v_to = float(np.trapz(a, dx=dt))
h = (v_to * v_to) / (2 * g)

print(f"Take-off @ {t_an[to_idx]:.3f}s | v_to={v_to:.2f} m/s | height={h:.3f} m | mass={m:.1f} kg")





