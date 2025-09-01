"""Evolution of COM force over time"""


# external
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import sys
import os

# internal 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.com_force as com_force
import src.utils as utils


def plot_evolution(timeseries, label, fig, ax):
    ax.plot(timeseries, label=label)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Force (N)')


if __name__ == "__main__": # this only runs when the script is executed directly
    cutoff_freq = 6.0  # Hz
    body_mass = 54.0  # kg, assumed body mass for the whole body COM calculation
    fs = 100 # Hz

    filepath = "/Users/adamdray/Downloads/Harriet_c3d/CMJ-002/pose_filt_0.c3d"
    cmj_1, labels_rotation = utils.load_data(filepath)
    matrices_dict = utils.extract_matrices(cmj_1, labels_rotation)
    positions = utils.extract_positions_from_matrices(matrices_dict)
    out = com_force.compute_whole_body_com_fixed(positions, body_mass, fs, cutoff_freq=6.0)

    # select z-component
    F_ext = out["F_ext"][:, 2]
    F_ext_smooth = out["F_ext_smooth"][:, 2]

    print(F_ext.shape)

    # === PLOTTING ===

    fig, ax = plt.subplots()
    plot_evolution(F_ext, 'Raw COM Force', fig, ax)
    plot_evolution(F_ext_smooth, 'Smooth COM Force', fig, ax)
    plt.title("Evolution of COM z-direction Force")
    plt.legend()
    plt.show()
