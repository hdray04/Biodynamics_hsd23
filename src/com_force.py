"""COM-based external force estimation utilities.

This module computes the whole-body center of mass (COM) from joint
positions and derives external force time series using Newton's laws.

Key function: `compute_whole_body_com_fixed` which takes a dict of joint
positions in millimetres, body mass (kg), and sampling frequency (Hz),
and returns filtered positions, velocities, accelerations, and raw/smoothed
external force estimates.

Units
- Input positions: millimetres (mm)
- Output COM positions: millimetres (mm)
- Output velocities/accelerations: derived from mm; where forces are computed
  accelerations are internally converted to m/s^2.
"""


# external
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
# internal
from . import utils
from typing import Dict, Any


def compute_whole_body_com_fixed(
    joints: Dict[str, np.ndarray],
    body_mass: float,
    fs: float,
    cutoff_freq: float = 6.0,
    g_vec: np.ndarray = np.array([0, 0, -9.81])
) -> Dict[str, Any]:
    """Compute whole-body COM and external forces from joint positions.

    Parameters:
    - joints: mapping of joint label -> (frames, 3) positions in mm.
      Required labels must match `src.utils.SEGMENTS` proximal/distal names.
    - body_mass: subject mass in kilograms.
    - fs: sampling frequency in Hz.
    - cutoff_freq: low-pass cutoff for COM position filtering (Hz).
    - g_vec: gravity vector in m/s^2 (default: [0, 0, -9.81]).

    Returns a dict containing:
    - r_com_raw: unfiltered COM positions (frames, 3) in mm
    - r_com: filtered COM positions (frames, 3) in mm
    - v_com: COM velocity from filtered position (mm/s)
    - a_com: COM acceleration from filtered position (mm/s^2)
    - F_ext: external force from raw acceleration (N)
    - F_ext_smooth: external force from filtered acceleration (N)
    - filter_info: metadata for the applied filter
    """
    segment_masses = []
    segment_coms = []

    for seg_name, (w_i, f_i, prox, dist) in utils.SEGMENTS.items():
        if prox not in joints or dist not in joints:
            raise KeyError(f"Missing joint: {prox} or {dist} in joints dict")

        r_prox = np.asarray(joints[prox])
        r_dist = np.asarray(joints[dist])
        r_seg_com = r_prox + f_i * (r_dist - r_prox)
        segment_coms.append(r_seg_com)
        segment_masses.append(w_i * body_mass)

    segment_coms = np.stack(segment_coms, axis=0)
    segment_masses = np.asarray(segment_masses)
    total_mass = np.sum(segment_masses)
    
    if not np.isclose(total_mass / body_mass, 1.0, atol=1e-3):
        raise ValueError("Segment mass fractions do not sum to 1.0")

    weighted = segment_coms * segment_masses[:, None, None]
    r_com_raw = np.sum(weighted, axis=0) / total_mass

    # Step 2: Apply low-pass filtering to position
    nyquist = fs / 2
    normalized_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(4, normalized_cutoff, 'low')
    r_com_filtered = signal.filtfilt(b, a, r_com_raw, axis=0)

    # Step 3: Calculate derivatives using central differences ON FILTERED DATA
    dt = 1.0 / fs
    def compute_derivatives(pos, dt):
        # Velocity using central differences
        v = np.zeros_like(pos)
        v[1:-1] = (pos[2:] - pos[:-2]) / (2 * dt)
        v[0] = (pos[1] - pos[0]) / dt
        v[-1] = (pos[-1] - pos[-2]) / dt

        # Acceleration using central differences
        a = np.zeros_like(v)
        a[1:-1] = (v[2:] - v[:-2]) / (2 * dt)
        a[0] = (v[1] - v[0]) / dt
        a[-1] = (v[-1] - v[-2]) / dt
        return v, a

    v_com, a_com = compute_derivatives(r_com_filtered, dt) # <- smooth one
    v_com_raw, a_com_raw = compute_derivatives(r_com_raw, dt)

    # Step 4: Calculate forces from the properly calculated acceleration
    # Convert acceleration to m/s² (it's currently in mm/s²)
    a_com_ms2 = a_com / 1000.0
    a_com_raw_ms2 = a_com_raw / 1000.0
    F_ext = total_mass * (a_com_raw_ms2 - g_vec[None, :])
    F_ext_smooth = total_mass * (a_com_ms2 - g_vec[None, :])
    
    return {
        "r_com_raw": r_com_raw,
        "r_com": r_com_filtered,
        "v_com": v_com,
        "a_com": a_com,
        "F_ext": F_ext,
        "F_ext_smooth": F_ext_smooth,
        "filter_info": {"cutoff": cutoff_freq, "fs": fs}
    }


if __name__ == "__main__":

    cutoff_freq = 6.0  # Hz
    body_mass = 65  # kg, assumed body mass for the whole body COM calculation
    fs = 100 # Hz

    filepath = "/Users/harrietdray/Library/CloudStorage/OneDrive-ImperialCollegeLondon/ACL_data/Pilot - Tash/Pilot - Tash_c3d - Sorted/Tash_SLDJ1_right/Take 2025-09-12 01-49-57 PM-016/pose_filt_0.c3d"
    cmj_1, labels_rotation = utils.load_data(filepath)
    matrices_dict = utils.extract_matrices(cmj_1, labels_rotation)
    positions = utils.extract_positions_from_matrices(matrices_dict)
    out = compute_whole_body_com_fixed(positions, body_mass, fs, cutoff_freq=6.0)

    r_com = out["r_com"]
    v_com = out["v_com"] 
    a_com = out["a_com"]
    F_ext = out["F_ext"]
    F_ext_smooth = out["F_ext_smooth"]

    # === PRINT RESULTS ===

    print("=== INPUTS ===")
    print(f"Sampling frequency: {fs} Hz")
    print(f"Body mass: {body_mass} kg")


    print("=== RESULTS ===")
    print(f"Position range (mm): {np.max(r_com[:,2]) - np.min(r_com[:,2]):.1f}")
    print(f"Jump height (COM peak - standing): {np.max(r_com[:,2]) - np.median(r_com[:100,2]):.3f} m")
    print(f"Max velocity (m/s): {np.max(v_com[:,2]) / 1000:.3f}")
   
    print(f"Max acceleration (m/s²): {np.max(a_com[:,2]) / 1000:.3f}")
    print(f"Max force (N): ")
    print(f"\t{np.max(F_ext[:,2]):.1f} (raw)")
    print(f"\t{np.max(F_ext_smooth[:,2]):.1f} (smooth)")
    print(f"Max bodyweight normalised force: ")
    print(f"\t{np.max(F_ext[:,2])/(body_mass*9.81):.1f} (raw)")
    print(f"\t{np.max(F_ext_smooth[:,2])/(body_mass*9.81):.1f} (smooth)")
