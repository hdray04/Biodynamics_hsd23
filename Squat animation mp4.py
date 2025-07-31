import ezc3d
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')  # Since this worked in the test
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import proj3d
from scipy.signal import find_peaks

# Load your data
cmj_2 = ezc3d.c3d("/Users/harrietdray/Biodynamics/Harriet_c3d/CMJ-002/pose_filt_0.c3d")
labels_rotation = cmj_2['parameters']['ROTATION']['LABELS']['value']
labels = cmj_2['parameters']['POINT']['LABELS']['value']

def extract_angles(cmj_2, labels):
    angle_data_trials = [cmj_2['data']['points']]
    angle_indices = {
        'left_knee': labels.index('LeftKneeAngles_Theia'),
        'right_knee': labels.index('RightKneeAngles_Theia'),
        'left_hip': labels.index('LeftHipAngles_Theia'),
        'right_hip': labels.index('RightHipAngles_Theia'),
        'left_fp': labels.index('LeftFootProgressionAngles_Theia'),
        'right_fp': labels.index('RightFootProgressionAngles_Theia'),
        'left_ankle': labels.index('LeftAnkleAngles_Theia'),
        'right_ankle': labels.index('RightAnkleAngles_Theia')
    }

    angle_data = angle_data_trials[0]
    n_frames = angle_data.shape[2]
    
    left_ankle_angles = angle_data[0, angle_indices['left_ankle'], :n_frames]
    right_ankle_angles = angle_data[0, angle_indices['right_ankle'], :n_frames]

    return {
        'left': left_ankle_angles,
        'right': right_ankle_angles,
        'frames': np.arange(n_frames),
        'n_frames': n_frames
    }

def extract_matrices_final(cmj_2, labels_rotation):
    rotation_data = cmj_2['data']['rotations']
    matrices_dict = {}
    n_joints = rotation_data.shape[2]
    
    for joint_idx in range(n_joints):
        if joint_idx < len(labels_rotation):
            label = labels_rotation[joint_idx]
            joint_name = label.replace('_4X4', '')
            joint_matrices = rotation_data[:, :, joint_idx, :].transpose(2, 0, 1)
            matrices_dict[joint_name] = joint_matrices
            
    return matrices_dict

def extract_positions_from_matrices(matrices_dict):
    positions = {}
    for joint_name, matrices in matrices_dict.items():
        positions[joint_name] = matrices[:, :3, 3]
    return positions

def create_interactive_cmj_viewer(positions, ankle_angles):

    
    # Create figure with space for slider
    fig = plt.figure(figsize=(16, 12))
    
    # Main 3D plot
    ax = plt.subplot2grid((20, 1), (0, 0), rowspan=18, projection='3d')
    
    # Slider area
    slider_ax = plt.subplot2grid((20, 1), (19, 0))
    slider_ax.set_xticks([])
    slider_ax.set_yticks([])

    # Define connections
    connections = [
        ('pelvis', 'torso'), ('torso', 'head'),
        ('pelvis', 'l_thigh'), ('l_thigh', 'l_shank'), ('l_shank', 'l_foot'),
        ('pelvis', 'r_thigh'), ('r_thigh', 'r_shank'), ('r_shank', 'r_foot'),
        ('torso', 'l_uarm'), ('l_uarm', 'l_larm'), ('l_larm', 'l_hand'),
        ('torso', 'r_uarm'), ('r_uarm', 'r_larm'), ('r_larm', 'r_hand'),
    ]

    # Colors
    colors = {
        'pelvis': 'red', 'torso': 'darkred', 'head': 'crimson',
        'l_thigh': 'blue', 'l_shank': 'lightblue', 'l_foot': 'navy',
        'r_thigh': 'green', 'r_shank': 'lightgreen', 'r_foot': 'darkgreen',
        'l_uarm': 'purple', 'l_larm': 'mediumpurple', 'l_hand': 'darkviolet',
        'r_uarm': 'orange', 'r_larm': 'gold', 'r_hand': 'darkorange',
    }

    # Initialize plot elements
    joint_scatters = {}
    connection_lines = []
    
    # Plot initial frame
    frame_idx = 0
    
    # Plot joints
    for joint_name, pos_data in positions.items():
        if len(pos_data) > frame_idx:  # Safety check
            pos = pos_data[frame_idx]
            color = colors.get(joint_name, 'gray')
            joint_scatters[joint_name] = ax.scatter(pos[0], pos[1], pos[2], 
                                                   s=120, c=color, alpha=0.9, 
                                                   edgecolors='black', linewidth=1)

    # Plot skeleton connections
    for joint1, joint2 in connections:
        if joint1 in positions and joint2 in positions:
            pos1 = positions[joint1][frame_idx]
            pos2 = positions[joint2][frame_idx]
            line, = ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]],
                           'k-', alpha=0.8, linewidth=3)
            connection_lines.append((line, joint1, joint2))

    # Initialize ankle label variables
    ankle_text_left = None
    ankle_text_right = None

    def update_ankle_labels(frame_idx):
        """Update horizontal ankle labels"""
        nonlocal ankle_text_left, ankle_text_right
        
        # Remove old labels
        if ankle_text_left is not None:
            ankle_text_left.remove()
        if ankle_text_right is not None:
            ankle_text_right.remove()
        
        # Add new horizontal labels in fixed screen positions
        ankle_text_left = ax.text2D(0.02, 0.95, 
                                  f"Left ankle: {ankle_angles['left'][frame_idx]:.1f}°", 
                                  transform=ax.transAxes, fontsize=14, color='blue', 
                                  fontweight='bold',
                                  bbox=dict(boxstyle="round,pad=0.4", facecolor="lightblue", alpha=0.9))
        
        ankle_text_right = ax.text2D(0.02, 0.88, 
                                   f"Right ankle: {ankle_angles['right'][frame_idx]:.1f}°", 
                                   transform=ax.transAxes, fontsize=14, color='green', 
                                   fontweight='bold',
                                   bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgreen", alpha=0.9))

    # Initialize labels
    update_ankle_labels(0)
    
    # Frame counter
    frame_text = ax.text2D(0.02, 0.81, f"Frame: 0/{len(positions['pelvis'])-1}", 
                          transform=ax.transAxes, fontsize=12, color='black',
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # Set up the plot
    ax.set_xlabel('X (mm)', fontsize=12)
    ax.set_ylabel('Y (mm)', fontsize=12)
    ax.set_zlabel('Z (mm)', fontsize=12)
    ax.set_title('Interactive cmj Viewer - Drag Slider to Navigate', fontsize=14, fontweight='bold')
    ax.set_box_aspect([1, 1, 1])

    # Set axis limits
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

    # Create slider
    n_frames = len(positions['pelvis'])
    slider = Slider(slider_ax, 'Frame', 0, n_frames-1, valinit=0, valstep=1, valfmt='%d')
    

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

        # Update ankle labels
        update_ankle_labels(frame_idx)
        
        # Update frame counter
        frame_text.set_text(f"Frame: {frame_idx}/{n_frames-1}")
        
        # Redraw
        fig.canvas.draw_idle()

    # Connect slider
    slider.on_changed(update_frame)
    
    # Keyboard controls
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
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    
    # Instructions
    instructions = ("Controls: Drag slider • Arrow keys: ←/→ = ±1 frame, ↑/↓ = ±10 frames")
    fig.text(0.5, 0.02, instructions, ha='center', fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    
    print("Showing interactive viewer...")
    plt.show(block=True)  # Keep window open
    
    return fig, slider

# Extract data
print("=== EXTRACTING cmj DATA ===")
ankle_angles = extract_angles(cmj_2, labels)
matrices_dict = extract_matrices_final(cmj_2, labels_rotation)
positions = extract_positions_from_matrices(matrices_dict)

print(f"Successfully extracted data for {len(positions['pelvis'])} frames")

# Create interactive viewer
if __name__ == "__main__":
    print("=== CREATING INTERACTIVE VIEWER ===")
    fig, slider = create_interactive_cmj_viewer(positions, ankle_angles)
    print("Interactive viewer completed")



plt.show()