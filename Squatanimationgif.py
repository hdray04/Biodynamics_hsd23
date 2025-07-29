import ezc3d
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import proj3d

# Load your data
squat_2 = ezc3d.c3d("/Users/harrietdray/Biodynamics/Harriet_c3d/Squat001-000/pose_filt_0.c3d")
labels_rotation = squat_2['parameters']['ROTATION']['LABELS']['value']
labels = squat_2['parameters']['POINT']['LABELS']['value']

def extract_knee_angles(squat_2, labels):
    angle_data_trials = [squat_2['data']['points']]
    angle_indices = {
        'left_knee': labels.index('LeftKneeAngles_Theia'),
        'right_knee': labels.index('RightKneeAngles_Theia'),
        'left_hip': labels.index('LeftHipAngles_Theia'),
        'right_hip': labels.index('RightHipAngles_Theia'),
        'left_fp': labels.index('LeftFootProgressionAngles_Theia'),
        'right_fp': labels.index('RightFootProgressionAngles_Theia')
    }

    angle_data = angle_data_trials[0]
    n_frames = angle_data.shape[2]
    
    left_knee_angles = -angle_data[0, angle_indices['left_knee'], :n_frames]
    right_knee_angles = angle_data[0, angle_indices['right_knee'], :n_frames]

    return {
        'left': left_knee_angles,
        'right': right_knee_angles,
        'frames': np.arange(n_frames),
        'n_frames': n_frames
    }

def extract_matrices_final(squat_2, labels_rotation):
    rotation_data = squat_2['data']['rotations']
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

def create_gif_with_slider(positions, knee_angles, filename='squat_with_slider.gif', 
                          fps=15, skip_frames=2):
    """Create an animated GIF with moving slider using Pillow"""
    
    print(f"Creating GIF with slider: {filename}")
    
    # Create figure with slider layout (same as your interactive version)
    fig = plt.figure(figsize=(16, 12))
    
    # Main 3D plot
    ax = plt.subplot2grid((20, 1), (0, 0), rowspan=17, projection='3d')
    
    # Slider area
    slider_ax = plt.subplot2grid((20, 1), (18, 0))
    
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

    # Plot initial joints
    for joint_name, pos_data in positions.items():
        pos = pos_data[0]
        color = colors.get(joint_name, 'gray')
        joint_scatters[joint_name] = ax.scatter(pos[0], pos[1], pos[2], 
                                               s=120, c=color, alpha=0.9, 
                                               edgecolors='black', linewidth=1)

    # Plot initial connections
    for joint1, joint2 in connections:
        if joint1 in positions and joint2 in positions:
            pos1 = positions[joint1][0]
            pos2 = positions[joint2][0]
            line, = ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]],
                           'k-', alpha=0.8, linewidth=3)
            connection_lines.append((line, joint1, joint2))

    # Set up the main plot
    ax.set_xlabel('X (mm)', fontsize=12)
    ax.set_ylabel('Y (mm)', fontsize=12)
    ax.set_zlabel('Z (mm)', fontsize=12)
    ax.set_title('Squat Motion Analysis with Moving Slider', fontsize=14, fontweight='bold')
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

    # Initialize text elements
    knee_text_left = ax.text2D(0.02, 0.95, 
                              f"Left Knee: {knee_angles['left'][0]:.1f}°", 
                              transform=ax.transAxes, fontsize=14, color='blue', 
                              fontweight='bold',
                              bbox=dict(boxstyle="round,pad=0.4", facecolor="lightblue", alpha=0.9))
    
    knee_text_right = ax.text2D(0.02, 0.88, 
                               f"Right Knee: {knee_angles['right'][0]:.1f}°", 
                               transform=ax.transAxes, fontsize=14, color='green', 
                               fontweight='bold',
                               bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgreen", alpha=0.9))
    
    frame_text = ax.text2D(0.02, 0.81, f"Frame: 0/{n_frames-1}", 
                          transform=ax.transAxes, fontsize=12, color='black',
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # Add instructions text
    instructions = ("Animated GIF showing squat motion with slider position")
    fig.text(0.5, 0.02, instructions, ha='center', fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    # Layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)

    def update_animation(frame_num):
        """Update function for animation - moves slider and updates skeleton"""
        frame_idx = frame_num * skip_frames
        if frame_idx >= n_frames:
            frame_idx = n_frames - 1
        
        # Update slider position (this makes the slider handle move!)
        slider.set_val(frame_idx)
        
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

        # Update text labels
        knee_text_left.set_text(f"Left Knee: {knee_angles['left'][frame_idx]:.1f}°")
        knee_text_right.set_text(f"Right Knee: {knee_angles['right'][frame_idx]:.1f}°")
        frame_text.set_text(f"Frame: {frame_idx}/{n_frames-1}")
        
        return []

    # Calculate animation frames
    anim_frames = n_frames // skip_frames
    
    # Create animation
    print(f"Creating animation with {anim_frames} frames at {fps} FPS...")
    anim = FuncAnimation(fig, update_animation, frames=anim_frames, 
                        interval=1000//fps, blit=False, repeat=True)

    # Save as GIF using Pillow
    print(f"Saving GIF to {filename}...")
    try:
        writer = PillowWriter(fps=fps)
        anim.save(filename, writer=writer, dpi=80)
        print(f"✅ GIF saved successfully as {filename}!")
        print(f"📁 File location: {filename}")
        
        # Get file size
        import os
        file_size = os.path.getsize(filename) / (1024 * 1024)  # Convert to MB
        print(f"📊 File size: {file_size:.1f} MB")
        
    except Exception as e:
        print(f"❌ Error saving GIF: {e}")
        print("Make sure pillow is installed: pip install pillow")
    
    plt.close(fig)
    return anim

# Extract data
print("=== EXTRACTING SQUAT DATA ===")
knee_angles = extract_knee_angles(squat_2, labels)
matrices_dict = extract_matrices_final(squat_2, labels_rotation)
positions = extract_positions_from_matrices(matrices_dict)

print(f"Successfully extracted data for {len(positions['pelvis'])} frames")

# Create GIF animations
if __name__ == "__main__":
    print("=== CREATING GIF ANIMATIONS ===")
    
    # First, make sure pillow is available
    try:
        import PIL
        print("✅ Pillow is available for GIF creation")
    except ImportError:
        print("❌ Pillow not found. Installing...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'pillow'])
        print("✅ Pillow installed successfully")
    
    # Create different quality GIFs
    
    # High quality (slower, larger file)
    print("\n🎬 Creating high quality GIF...")
    create_gif_with_slider(positions, knee_angles, 
                          filename='squat_high_quality.gif', 
                          fps=20,         # 20 FPS
                          skip_frames=1)  # Use all frames
    
    # Medium quality (good balance)
    print("\n🎬 Creating medium quality GIF...")
    create_gif_with_slider(positions, knee_angles, 
                          filename='squat_medium_quality.gif', 
                          fps=15,         # 15 FPS
                          skip_frames=2)  # Every 2nd frame
    
    
    # Low quality preview (fast, small file)
    print("\n🎬 Creating preview GIF...")
    create_gif_with_slider(positions, knee_angles, 
                          filename='squat_preview.gif', 
                          fps=10,         # 10 FPS
                          skip_frames=4)  # Every 4th frame
    
    print("\n🎉 All GIFs created successfully!")
    print("\n📁 Files created:")
    print("   • squat_high_quality.gif (20 FPS, all frames)")
    print("   • squat_medium_quality.gif (15 FPS, every 2nd frame)")
    print("   • squat_preview.gif (10 FPS, every 4th frame)")
    print("\n💡 You can open these GIFs in any web browser, image viewer, or share them anywhere!")