"""Combined 3D + evolution plot (with COM)"""

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
from plotting.threeD_plot import init_3d_artists, update_3d_frame, attach_slider_and_keys


def plot_evolution(ax, t, series_list, labels):
    for s, lab in zip(series_list, labels):
        ax.plot(t, s, label=lab, linewidth=1.5)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Force (N)', fontsize=12)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Format y-axis to avoid scientific notation crowding
    ax.ticklabel_format(style='plain', axis='y')
    
    # Cursor line to show current frame over time
    cursor = ax.axvline(t[0], color='red', linestyle='--', alpha=0.8, linewidth=2)
    return cursor


if __name__ == "__main__":  # this only runs when the script is executed directly
    cutoff_freq = 6.0  # Hz
    body_mass = 54.0  # kg, assumed body mass for the whole body COM calculation
    fs = 100  # Hz

    filepath = "/Users/adamdray/Downloads/Harriet_c3d/CMJ-002/pose_filt_0.c3d"
    cmj_1, labels_rotation = utils.load_data(filepath)
    matrices_dict = utils.extract_matrices(cmj_1, labels_rotation)
    positions = utils.extract_positions_from_matrices(matrices_dict)
    out = com_force.compute_whole_body_com_fixed(positions, body_mass, fs, cutoff_freq=cutoff_freq)

    # Timeseries (z-component of external force)
    F_ext = out["F_ext"][:, 2]
    F_ext_smooth = out["F_ext_smooth"][:, 2]
    n = F_ext.shape[0]
    t = np.arange(n) / fs

    # COM positions for 3D
    com = out["r_com"]

    # === COMBINED PLOTTING ===

    fig = plt.figure(figsize=(16, 8))
    # 3D plot on left (40% width), force evolution on right (60% width) with proper spacing
    ax3d = plt.subplot2grid((12, 12), (0, 0), rowspan=7, colspan=4, projection='3d')
    ax_ts = plt.subplot2grid((12, 12), (0, 5), rowspan=7, colspan=7)

    # 3D scene
    artists = init_3d_artists(ax3d, positions, com)
    ax3d.set_title('3D Body Animation', fontsize=14, fontweight='bold', pad=20)

    # Time series
    cursor = plot_evolution(ax_ts, t, [F_ext, F_ext_smooth], ["Raw COM Force (Z)", "Smooth COM Force (Z)"])
    ax_ts.set_title("COM Force Over Time (Z-axis)")

    # Instructions and layout
    plt.subplots_adjust(bottom=0.20, left=0.05, right=0.98, top=0.90, wspace=0.15, hspace=0.3)
    fig.text(0.5, 0.02, "Controls: Drag slider • Arrow keys: ←/→ = ±1, ↑/↓ = ±10",
             ha='center', fontsize=11,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="lightblue", alpha=0.8, edgecolor='gray'))

    # Shared update
    def on_frame_change(idx):
        # Update 3D
        update_3d_frame(idx, artists, positions, com)
        # Update cursor
        x = t[min(max(idx, 0), n - 1)]
        cursor.set_xdata([x, x])
        # Update slider label to show current frame
        slider.label.set_text(f'Frame: {int(idx)}/{artists["n_frames"]-1}')
        fig.canvas.draw_idle()

    # Create full-width slider at bottom with more space and centered label
    slider_ax = plt.subplot2grid((12, 12), (8, 1), colspan=10)  # Center the slider wider
    slider_ax.set_xticks([])
    slider_ax.set_yticks([])
    slider = Slider(slider_ax, f'Frame: 0/{artists["n_frames"]-1}', 0, artists['n_frames']-1, valinit=0, valstep=1, valfmt='%d')
    
    # Make slider thicker and easier to use
    slider.poly.set_height(0.6)  # Make slider track thicker
    slider.vline.set_linewidth(4)  # Make slider handle thicker
    slider.vline.set_color('darkblue')  # Color the handle

    # Keyboard controls
    def on_key(event):
        current_frame = int(slider.val)
        if event.key == 'right' and current_frame < artists['n_frames'] - 1:
            slider.set_val(current_frame + 1)
        elif event.key == 'left' and current_frame > 0:
            slider.set_val(current_frame - 1)
        elif event.key == 'up' and current_frame < artists['n_frames'] - 10:
            slider.set_val(current_frame + 10)
        elif event.key == 'down' and current_frame >= 10:
            slider.set_val(current_frame - 10)

    slider.on_changed(lambda val: on_frame_change(int(val)))
    fig.canvas.mpl_connect('key_press_event', on_key)

    # Show
    plt.show()

