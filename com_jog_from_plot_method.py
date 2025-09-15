"""Compute COM-based vertical force for Jog-001 using the same method
as plotting/force_3d_combination_plot.py (utils + com_force).
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import ezc3d

# Internal imports (same pattern as force_3d_combination_plot.py)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import src.com_force as com_force
import src.utils as utils


def main():
    cutoff_freq = 6.0  # Hz
    body_mass = 54.0   # kg
    fs = 100           # Hz (match plotting script default)

    # Use the Jog-001 file path already used in the repo
    filepath = "/Users/harrietdray/Biodynamics/Harriet_c3d/Jog-001/pose_filt_0.c3d"

    # Load and extract positions via the same utilities
    c3d_obj, labels_rotation = utils.load_data(filepath)
    matrices_dict = utils.extract_matrices(c3d_obj, labels_rotation)
    positions = utils.extract_positions_from_matrices(matrices_dict)

    # Compute COM and external forces
    out = com_force.compute_whole_body_com_fixed(positions, body_mass, fs, cutoff_freq=cutoff_freq)

    F_ext = out["F_ext"][:, 2]
    F_ext_smooth = out["F_ext_smooth"][:, 2]
    n = F_ext.shape[0]
    t = np.arange(n) / fs

    # Summary
    bw = body_mass * 9.81
    print("=== COM/Force Summary (Jog-001) ===")
    print(f"Frames: {n}")
    print(f"Peak Fz (raw):    {np.max(F_ext):.1f} N  ({np.max(F_ext)/bw:.2f} BW)")
    print(f"Peak Fz (smooth): {np.max(F_ext_smooth):.1f} N  ({np.max(F_ext_smooth)/bw:.2f} BW)")

    # Simple force–time plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, F_ext, label='Raw COM Force (Z)')
    ax.plot(t, F_ext_smooth, label='Smooth COM Force (Z)')
    ax.axhline(bw, color='gray', ls='--', lw=1, label='1 BW')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Force (N)')
    ax.set_title('COM Force Over Time (Jog-001, Z-axis)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

