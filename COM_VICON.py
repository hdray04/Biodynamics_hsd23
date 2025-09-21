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


# Define segment metadata for lower body

lower_body_segments = [
    {"name": "pelvis",
     "proximal": ["LASI", "RASI", "LPSI", "RPSI"],  # mean of all four
     "distal":   None,
     "com_proportion": 0.0,  # Not used for pelvis
     "mass_fraction": 0.142},
    {"name": "left_thigh",
     "proximal": ["LASI", "RASI"],  # mean of two; hip
     "distal":   ["LKNE"],
     "com_proportion": 0.433,
     "mass_fraction": 0.100},
    {"name": "right_thigh",
     "proximal": ["LASI", "RASI"],
     "distal":   ["RKNE"],
     "com_proportion": 0.433,
     "mass_fraction": 0.100},
    {"name": "left_shank",
     "proximal": ["LKNE"],
     "distal":   ["LANK"],
     "com_proportion": 0.433,
     "mass_fraction": 0.0465},
    {"name": "right_shank",
     "proximal": ["RKNE"],
     "distal":   ["RANK"],
     "com_proportion": 0.433,
     "mass_fraction": 0.0465},
    {"name": "left_foot",
     "proximal": ["LHEE"],
     "distal":   ["LTOE"],
     "com_proportion": 0.5,
     "mass_fraction": 0.0145},
    {"name": "right_foot",
     "proximal": ["RHEE"],
     "distal":   ["RTOE"],
     "com_proportion": 0.5,
     "mass_fraction": 0.0145},
]

# Helper to get marker coordinates
def mean_markers(marker_list):
    idx = [labels.index(label) for label in marker_list]
    return np.mean(points[:3, idx, :], axis=1)  # (3, n_frames)

# Process segments in a loop
segment_COMs = []
segment_weights = []

for seg in lower_body_segments:
    prox = mean_markers(seg["proximal"])
    if seg["distal"] is None:  # For pelvis: just use marker mean
        seg_COM = prox  # No vector (pelvis "length" not used here)
    else:
        dist = mean_markers(seg["distal"])
        seg_COM = prox + seg["com_proportion"] * (dist - prox)
    segment_COMs.append(seg_COM)
    segment_weights.append(seg["mass_fraction"])

# Stack and weight
segment_COMs = np.stack(segment_COMs, axis=0)  # (n_segs, 3, n_frames)
segment_weights = np.array(segment_weights).reshape(-1, 1, 1)

# Compute lower body COM
lower_body_COM = np.sum(segment_COMs * segment_weights, axis=0) / np.sum(segment_weights)

# Extract Z (vertical) coordinate if needed
lower_body_COM_z = lower_body_COM[2, :]

print(f"Lower body COM Z range: {lower_body_COM_z.min():.2f} to {lower_body_COM_z.max():.2f} mm")

#Calculate jump height from COM
standing_z = np.mean(lower_body_COM_z[:200])
max_z = np.max(lower_body_COM_z)
jump_height = max_z - standing_z
print(f"Estimated jump height from lower body COM: {jump_height:.2f} mm")