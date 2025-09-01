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


def plot_3d(positions, com=None):
    """
        Interactive 3D plot of joint positions and, optionally,
        the center of mass (COM). Leave com as None to hide the COM.
    """

    fig = plt.figure(figsize=(16, 12))
    ax = plt.subplot2grid((20, 1), (0, 0), rowspan=18, projection='3d')

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
            pos = pos_data[frame_idx]
            color = colors.get(joint_name, 'gray')
            joint_scatters[joint_name] = ax.scatter(pos[0], pos[1], pos[2], 
                                                   s=120, c=color, alpha=0.9, 
                                                   edgecolors='black', linewidth=1)
    # Plot joint connections
    for joint1, joint2 in connections:
        if joint1 in positions and joint2 in positions:
            pos1 = positions[joint1][frame_idx]
            pos2 = positions[joint2][frame_idx]
            line, = ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]],
                           'k-', alpha=0.8, linewidth=3)
            connection_lines.append((line, joint1, joint2))

    if com is not None:
        com_pos = com[frame_idx]
        # Draw the COM as a black dot
        com_scatter = ax.scatter(com_pos[0], com_pos[1], com_pos[2], s=200, c='black', alpha=0.9,
               edgecolors='black', linewidth=1, label='COM')
        # Draw an X marker at the COM position
        com_x_marker = ax.scatter(com_pos[0], com_pos[1], com_pos[2], s=300, c='red', marker='x', linewidth=3)

    # === FORMATTING ===

    # Slider
    slider_ax = plt.subplot2grid((20, 1), (19, 0))
    slider_ax.set_xticks([])
    slider_ax.set_yticks([])
    n_frames = len(positions['pelvis'])
    slider = Slider(slider_ax, 'Frame', 0, n_frames-1, valinit=0, valstep=1, valfmt='%d')
    
    # Frame counter
    frame_text = ax.text2D(0.02, 0.81, f"Frame: 0/{len(positions['pelvis'])-1}", 
                          transform=ax.transAxes, fontsize=12, color='black',
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # Labels
    ax.set_xlabel('X (mm)', fontsize=12)
    ax.set_ylabel('Y (mm)', fontsize=12)
    ax.set_zlabel('Z (mm)', fontsize=12)
    ax.set_title('Interactive cmj Viewer - Drag Slider to Navigate', fontsize=14, fontweight='bold')
    ax.set_box_aspect([1, 1, 1])

    # Axis
    all_positions = np.concatenate([positions[joint] for joint in positions], axis=0)
    margin = 200
    x_range = [all_positions[:, 0].min() - margin, all_positions[:, 0].max() + margin]
    y_range = [all_positions[:, 1].min() - margin, all_positions[:, 1].max() + margin]
    z_range = [all_positions[:, 2].min() - margin, all_positions[:, 2].max() + margin]
    max_range = max(x_range[1] - x_range[0], y_range[1] - y_range[0], z_range[1] - z_range[0])
    center = [(x_range[0] + x_range[1])/2, (y_range[0] + y_range[1])/2, (z_range[0] + z_range[1])/2]
    ax.set_xlim([center[0] - max_range/2, center[0] + max_range/2])
    ax.set_ylim([center[1] - max_range/2, center[1] + max_range/2])
    ax.set_zlim([center[2] - max_range/2, center[2] + max_range/2])

    # Other 
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    instructions = ("Controls: Drag slider • Arrow keys: ←/→ = ±1 frame, ↑/↓ = ±10 frames")
    fig.text(0.5, 0.02, instructions, ha='center', fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    
    # === ANIMATION ===

    def update_frame(val):
        """Update function for slider"""
        frame_idx = int(slider.val)
        
        # Update joint positions
        for joint_name, scatter in joint_scatters.items():
            if joint_name in positions and frame_idx < len(positions[joint_name]):
                pos = positions[joint_name][frame_idx]
                scatter._offsets3d = ([pos[0]], [pos[1]], [pos[2]])

        # Update connection lines
        for line, joint1, joint2 in connection_lines:
            if (joint1 in positions and joint2 in positions and 
                frame_idx < len(positions[joint1]) and frame_idx < len(positions[joint2])):
                pos1 = positions[joint1][frame_idx]
                pos2 = positions[joint2][frame_idx]
                line.set_data([pos1[0], pos2[0]], [pos1[1], pos2[1]])
                line.set_3d_properties([pos1[2], pos2[2]])
        
        # Update COM position
        if com is not None and com_scatter is not None and com_x_marker is not None and frame_idx < len(com):
            com_pos = com[frame_idx]
            com_scatter._offsets3d = ([com_pos[0]], [com_pos[1]], [com_pos[2]])
            com_x_marker._offsets3d = ([com_pos[0]], [com_pos[1]], [com_pos[2]])
        
        # Update frame counter
        frame_text.set_text(f"Frame: {frame_idx}/{n_frames-1}")
        
        # Update the plot
        fig.canvas.draw_idle()

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
    

    slider.on_changed(update_frame) # control by slider
    fig.canvas.mpl_connect('key_press_event', on_key) # control by keys

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