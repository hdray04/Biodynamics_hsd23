'''3D plot of COM with existing points'''


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


def _compute_axis_limits(positions, margin=200):
    """Compute finite axis limits robustly; fall back if data invalid."""
    try:
        stacks = []
        for joint, arr in positions.items():
            a = np.asarray(arr)
            if a.ndim != 2 or a.shape[1] < 3:
                continue
            mask = np.isfinite(a).all(axis=1)
            if np.any(mask):
                stacks.append(a[mask, :3])
        if not stacks:
            raise ValueError('No finite positions')
        all_positions = np.concatenate(stacks, axis=0)
        x0, x1 = float(np.nanmin(all_positions[:, 0]) - margin), float(np.nanmax(all_positions[:, 0]) + margin)
        y0, y1 = float(np.nanmin(all_positions[:, 1]) - margin), float(np.nanmax(all_positions[:, 1]) + margin)
        z0, z1 = float(np.nanmin(all_positions[:, 2]) - margin), float(np.nanmax(all_positions[:, 2]) + margin)
        max_range = max(x1 - x0, y1 - y0, z1 - z0)
        if not np.isfinite(max_range) or max_range <= 0:
            raise ValueError('Non-positive range')
        center = [(x0 + x1) / 2.0, (y0 + y1) / 2.0, (z0 + z1) / 2.0]
        return center, max_range
    except Exception:
        # Fallback to a reasonable cube around origin
        return [0.0, 0.0, 0.0], 1000.0


def init_3d_artists(ax, positions, com=None):
    """
    Add data and styles to 3D axes and
    return a dict of artists and state for later updates.
    """

    # === PLOTTING ===

    # Joint adjacency
    connections = [
        ('pelvis', 'torso'), ('torso', 'head'),
        ('pelvis', 'l_thigh'), ('l_thigh', 'l_shank'), ('l_shank', 'l_foot'),
        ('pelvis', 'r_thigh'), ('r_thigh', 'r_shank'), ('r_shank', 'r_foot'),
        ('torso', 'l_uarm'), ('l_uarm', 'l_larm'), ('l_larm', 'l_hand'),
        ('torso', 'r_uarm'), ('r_uarm', 'r_larm'), ('r_larm', 'r_hand'),
    ]

    # Joint Colors
    colors = {
        'pelvis': 'red', 'torso': 'darkred', 'head': 'crimson',
        'l_thigh': 'blue', 'l_shank': 'lightblue', 'l_foot': 'navy',
        'r_thigh': 'green', 'r_shank': 'lightgreen', 'r_foot': 'darkgreen',
        'l_uarm': 'purple', 'l_larm': 'mediumpurple', 'l_hand': 'darkviolet',
        'r_uarm': 'orange', 'r_larm': 'gold', 'r_hand': 'darkorange',
    }

    joint_scatters = {}
    connection_lines = []
    com_scatter = None
    com_x_marker = None
    frame_idx = 0
    
    # Plot joints positions
    for joint_name, pos_data in positions.items():
        if len(pos_data) > frame_idx:  # Safety check
            pos = np.asarray(pos_data[frame_idx])
            if pos.shape[0] >= 3 and np.all(np.isfinite(pos[:3])):
                color = colors.get(joint_name, 'gray')
                joint_scatters[joint_name] = ax.scatter(pos[0], pos[1], pos[2], 
                                                       s=120, c=color, alpha=0.9, 
                                                       edgecolors='black', linewidth=1)
    # Plot joint connections
    for joint1, joint2 in connections:
        if joint1 in positions and joint2 in positions:
            pos1 = np.asarray(positions[joint1][frame_idx])
            pos2 = np.asarray(positions[joint2][frame_idx])
            if pos1.shape[0] >= 3 and pos2.shape[0] >= 3 and np.all(np.isfinite(pos1[:3])) and np.all(np.isfinite(pos2[:3])):
                line, = ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]],
                               'k-', alpha=0.8, linewidth=3)
                connection_lines.append((line, joint1, joint2))

    if com is not None and len(com) > frame_idx:
        com_pos = np.asarray(com[frame_idx])
        if np.all(np.isfinite(com_pos[:3])):
            # Draw the COM as a black dot
            com_scatter = ax.scatter(com_pos[0], com_pos[1], com_pos[2], s=200, c='black', alpha=0.9,
                   edgecolors='black', linewidth=1, label='COM')
            # Draw an X marker at the COM position
            com_x_marker = ax.scatter(com_pos[0], com_pos[1], com_pos[2], s=300, c='red', marker='x', linewidth=3)
    
    # Labels
    ax.set_xlabel('X (mm)', fontsize=12)
    ax.set_ylabel('Y (mm)', fontsize=12)
    ax.set_zlabel('Z (mm)', fontsize=12)
    # ax.set_title('Interactive cmj Viewer - Drag Slider to Navigate', fontsize=14, fontweight='bold')
    ax.set_box_aspect([1, 1, 1])

    # Axis (robust)
    center, max_range = _compute_axis_limits(positions)
    ax.set_xlim([center[0] - max_range/2, center[0] + max_range/2])
    ax.set_ylim([center[1] - max_range/2, center[1] + max_range/2])
    ax.set_zlim([center[2] - max_range/2, center[2] + max_range/2])

    # Frame counter
    # Determine available frame count robustly
    if positions:
        n_frames = int(min(len(v) for v in positions.values() if hasattr(v, '__len__') and len(v) > 0))
    else:
        n_frames = 0
    frame_text = ax.text2D(0.02, 0.81, f"Frame: 0/{n_frames-1}", 
                           transform=ax.transAxes, fontsize=12, color='black',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    return {
        'joint_scatters': joint_scatters,
        'connection_lines': connection_lines,
        'com_scatter': com_scatter,
        'com_x_marker': com_x_marker,
        'frame_text': frame_text,
        'n_frames': n_frames,
    }


def update_3d_frame(frame_idx, artists, positions, com=None):
    """Update joint, connections, and optional COM to a frame index."""
    # Update joint positions
    for joint_name, scatter in artists['joint_scatters'].items():
        if joint_name in positions and frame_idx < len(positions[joint_name]):
            pos = positions[joint_name][frame_idx]
            scatter._offsets3d = ([pos[0]], [pos[1]], [pos[2]])

    # Update connection lines
    for line, joint1, joint2 in artists['connection_lines']:
        if (joint1 in positions and joint2 in positions and 
            frame_idx < len(positions[joint1]) and frame_idx < len(positions[joint2])):
            pos1 = positions[joint1][frame_idx]
            pos2 = positions[joint2][frame_idx]
            line.set_data([pos1[0], pos2[0]], [pos1[1], pos2[1]])
            line.set_3d_properties([pos1[2], pos2[2]])

    # Update COM position
    if com is not None and artists['com_scatter'] is not None and artists['com_x_marker'] is not None and frame_idx < len(com):
        com_pos = com[frame_idx]
        artists['com_scatter']._offsets3d = ([com_pos[0]], [com_pos[1]], [com_pos[2]])
        artists['com_x_marker']._offsets3d = ([com_pos[0]], [com_pos[1]], [com_pos[2]])

    # Update frame counter
    artists['frame_text'].set_text(f"Frame: {frame_idx}/{artists['n_frames']-1}")


def attach_slider_and_keys(fig, n_frames, on_change):
    """Add a bottom slider and arrow-key controls to the figure."""
    slider_ax = plt.subplot2grid((20, 1), (19, 0))
    slider_ax.set_xticks([])
    slider_ax.set_yticks([])
    slider = Slider(slider_ax, 'Frame', 0, n_frames-1, valinit=0, valstep=1, valfmt='%d')

    def on_key(event):
        current_frame = int(slider.val)
        if event.key == 'right' and current_frame < n_frames - 1:
            slider.set_val(current_frame + 1)
        elif event.key == 'left' and current_frame > 0:
            slider.set_val(current_frame - 1)
        elif event.key == 'up' and current_frame < n_frames - 10:
            slider.set_val(current_frame + 10)
        elif event.key == 'down' and current_frame >= 10:
            slider.set_val(current_frame - 10)

    slider.on_changed(lambda val: on_change(int(val)))
    fig.canvas.mpl_connect('key_press_event', on_key)
    return slider


def plot_3d(positions, com=None):
    """
        Interactive 3D plot of joint positions and, optionally,
        the center of mass (COM). Leave com as None to hide the COM.
    """

    fig = plt.figure(figsize=(14, 10))
    ax = plt.subplot2grid((20, 1), (0, 0), rowspan=18, projection='3d')

    artists = init_3d_artists(ax, positions, com)

    # Other 
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    instructions = ("Controls: Drag slider • Arrow keys: ←/→ = ±1 frame, ↑/↓ = ±10 frames")
    fig.text(0.5, 0.02, instructions, ha='center', fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    # Controls
    slider = attach_slider_and_keys(fig, artists['n_frames'], lambda idx: (update_3d_frame(idx, artists, positions, com), fig.canvas.draw_idle()))

    # Show the plot
    plt.show(block=True)  # Keep window open

    return fig, slider


if __name__ == "__main__": # this only runs when the script is executed directly
    filepath = "/Users/adamdray/Downloads/Harriet_c3d/CMJ-002/pose_filt_0.c3d"
    cmj_1, labels_rotation = utils.load_data(filepath)
    matrices_dict = utils.extract_matrices(cmj_1, labels_rotation)
    positions = utils.extract_positions_from_matrices(matrices_dict)

    body_mass = 54 # kg
    com_calcs = com_force.compute_whole_body_com_fixed(positions, body_mass, fs=100, cutoff_freq=6.0)
    com = com_calcs["r_com"]

    plot_3d(positions, com)
