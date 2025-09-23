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
    body_mass = 65  # kg, assumed body mass for the whole body COM calculation
    fs = 100  # Hz

    #filepath = '/Users/harrietdray/Library/CloudStorage/OneDrive-ImperialCollegeLondon/ACL_data/Pilot - Tash/Pilot - Tash_c3d - Sorted/Tash_SLDJ2_left/Take 2025-09-12 01-49-57 PM-014/pose_filt_0.c3d'
    #filepath = "/Users/harrietdray/Library/CloudStorage/OneDrive-ImperialCollegeLondon/ACL_data/Pilot - Tash/Pilot - Tash_c3d - Sorted/SLDJ1_right/Take 2025-09-12 01-49-57 PM-015/pose_filt_0.c3d"
    #filepath = '/Users/harrietdray/Library/CloudStorage/OneDrive-ImperialCollegeLondon/ACL_data/Pilot - Tash/Pilot - Tash_c3d - Sorted/Tash_SLDJ1_right/Take 2025-09-12 01-49-57 PM-016/pose_filt_0.c3d'
    #filepath = '/Users/harrietdray/Library/CloudStorage/OneDrive-ImperialCollegeLondon/ACL_data/Pilot - Tash/Pilot - Tash_c3d - Sorted/SLDJ3_right/Take 2025-09-12 01-49-57 PM-017/pose_filt_0.c3d'
    #filepath = '/Users/harrietdray/Library/CloudStorage/OneDrive-ImperialCollegeLondon/ACL_data/Pilot - Tash/Pilot - Tash_c3d - Sorted/single_hop_left_1/Take 2025-09-12 01-49-57 PM-022/pose_filt_0.c3d'
    #filepath = '/Users/harrietdray/Library/CloudStorage/OneDrive-ImperialCollegeLondon/ACL_data/Pilot - Tash/Pilot - Tash_c3d - Sorted/Tash_single_hop_left_3/Take 2025-09-12 01-49-57 PM-023/pose_filt_0.c3d'
    #filepath = '/Users/harrietdray/Library/CloudStorage/OneDrive-ImperialCollegeLondon/ACL_data/Pilot - Tash/Pilot - Tash_c3d - Sorted/Take 2025-09-12 01-49-57 PM-021-single_hop_left/pose_filt_0.c3d'
    #filepath = '/Users/harrietdray/Library/CloudStorage/OneDrive-ImperialCollegeLondon/ACL_data/Pilot - Tash/Pilot - Tash_c3d - Sorted/single_hop_left_2/pose_filt_0.c3d'
    #filepath = '/Users/harrietdray/Library/CloudStorage/OneDrive-ImperialCollegeLondon/ACL_data/Pilot - Tash/Pilot - Tash_c3d - Sorted/Tash_single_hop_right_1/Take 2025-09-12 01-49-57 PM-024/pose_filt_0.c3d'
    #filepath = '/Users/harrietdray/Library/CloudStorage/OneDrive-ImperialCollegeLondon/ACL_data/Pilot - Tash/Pilot - Tash_c3d - Sorted/single_hop_right_1/Take 2025-09-12 01-49-57 PM-023/pose_filt_0.c3d'
    filepath = '/Users/harrietdray/Library/CloudStorage/OneDrive-ImperialCollegeLondon/ACL_data/Pilot - Tash/Pilot - Tash_c3d - Sorted/Tash_single_hop_right_2/Take 2025-09-12 01-49-57 PM-025/pose_filt_0.c3d'

    cmj, labels_rotation = utils.load_data(filepath)
    matrices_dict = utils.extract_matrices(cmj, labels_rotation)
    positions = utils.extract_positions_from_matrices(matrices_dict)
    out = com_force.compute_whole_body_com_fixed(positions, body_mass, fs, cutoff_freq=cutoff_freq)

    # Timeseries (z-component of external force)
    F_ext = out["F_ext"][:, 2]
    F_ext_smooth = out["F_ext_smooth"][:, 2]
    n = F_ext.shape[0]
    t = np.arange(n) / fs

    # COM positions for 3D
    com = out["r_com"]

    # --- Calculate change in X for left foot marker ---

    # 1. Find the index of the greatest peak in COM force (landing)
    landing_idx = np.argmax(F_ext_smooth)

    # 2. Get the left foot marker positions (assuming label contains 'LFOOT' or similar)
    foot_label = positions['r_foot']
    # Plot l_foot marker position in X, Y, Z axes
    foot_xyz = foot_label  # shape: (n_frames, 3)
    # Plot left foot marker position in X, Y, Z axes across frames
    fig_foot, axs_foot = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    axes_labels = ['X', 'Y', 'Z']
    for i in range(3):
        axs_foot[i].plot(np.arange(foot_xyz.shape[0]), foot_xyz[:, i], label=f'L_FOOT {axes_labels[i]}')
        axs_foot[i].set_ylabel(f'{axes_labels[i]} Position (mm)')
        axs_foot[i].legend(loc='upper right', fontsize=9)
        axs_foot[i].grid(True, alpha=0.3)
    axs_foot[2].set_xlabel('Frame')
    fig_foot.suptitle('Left Foot Marker Position Across Frames')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # Calculate average Y of left foot ~50 frames after landing
    post_landing_start = 210
    post_landing_end = 210 + 49
    post_landing_end = min(post_landing_end, foot_xyz.shape[0])  # avoid overflow
    avg_y_post_landing = np.mean(foot_xyz[post_landing_start:post_landing_end, 1])

    # Calculate median Y of left foot in first 100 frames
    start_frames = min(100, foot_xyz.shape[0])
    median_y_start = np.median(foot_xyz[:start_frames, 1])

    print(f"Average left foot Y (frames {post_landing_start}-{post_landing_end}): {avg_y_post_landing:.4f}")
    print(f"Median left foot Y (first {start_frames} frames): {median_y_start:.4f}")
    jump_distance = avg_y_post_landing - median_y_start
    print(f"Estimated jump distance (Y change): {jump_distance:.4f} mm")


    #COM jump height 
    # # For drop jump: jump height is max - min COM height between frames 200 and 270
    # com_window = com[160:230, 2]
    # com_min = np.min(com_window)
    # com_max = np.max(com_window)
    # jump_height = com_max - com_min
    # print(f"Min COM Z: {com_min:.3f} m at frame {np.argmin(com_window)+160}, max COM Z: {com_max:.3f} m at frame {np.argmax(com_window)+160}")
    # print(f"Drop jump height (max - min COM between frames 200-300): {jump_height:.3f} mm")

    # # === SIMPLE PLOT: COM HEIGHT OVER TIME ===
    # plt.figure(figsize=(8, 4))
    # plt.plot(np.arange(com.shape[0]), com[:, 2], label='COM Height (Z)', color='purple')
    # plt.xlabel('Frames')
    # plt.ylabel('COM Height (m)')
    # plt.title('Center of Mass Height Over Time')
    # plt.grid(True, alpha=0.3)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
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
    # Add a vertical line at frame 351

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

