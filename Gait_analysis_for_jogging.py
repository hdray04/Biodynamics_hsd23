
import ezc3d
myjog = ezc3d.c3d("/Users/harrietdray/Biodynamics/Harriet_c3d/Jog-001/pose_filt_0.c3d")
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#Extract 4x4 transformatino matrices for each joint across data specified 
def extract_matrices_final(myjog, labels):
    rotation_data = myjog['data']['rotations']  # Shape: (4, 4, 19, 547)
    
    print(f"Rotation data shape: {rotation_data.shape}")
    print("Structure: [4x4 matrix, 19 joints, 547 frames]")
    
    matrices_dict = {} # Dictionary to hold matrices for each joint
    n_joints = rotation_data.shape[2]  # 19 joints
    n_frames = rotation_data.shape[3]  # 547 frames
    
    
    for joint_idx in range(n_joints):
        if joint_idx < len(labels):
            label = labels[joint_idx]
            joint_name = label.replace('_4X4', '')
            
            # Extract all matrices for this joint across all frames
            # Shape will be (4, 4, n_frames) -> we want (n_frames, 4, 4)
            joint_matrices = rotation_data[:, :, joint_idx, :].transpose(2, 0, 1)
            
            matrices_dict[joint_name] = joint_matrices
            
    print(f"Extracted {len(matrices_dict)} joints")
    return matrices_dict

def extract_positions_from_matrices(matrices_dict):
    positions = {}
    
    for joint_name, matrices in matrices_dict.items():
        # Extract translation component (last column, first 3 rows)
        # matrices shape: (n_frames, 4, 4)
        positions[joint_name] = matrices[:, :3, 3]  # (n_frames, 3)
    
    return positions

def extract_angles(myjog, labels2):
    angle_data = myjog['data']['points']  # Shape: (3, 19, 547)
    angle_indices = {
    'left_knee': labels2.index('LeftKneeAngles_Theia'),
    'right_knee': labels2.index('RightKneeAngles_Theia'),
    'left_hip': labels2.index('LeftHipAngles_Theia'),
    'right_hip': labels2.index('RightHipAngles_Theia'),
    'left_fp': labels2.index('LeftFootProgressionAngles_Theia'),
    'right_fp': labels2.index('RightFootProgressionAngles_Theia')
    }

    angles = {key: {'Sagittal': [], 'Frontal': [], 'Transverse': []} for key in angle_indices.keys()}
    
    n_frames = angle_data.shape[2]
    
    for key, idx in angle_indices.items():
        # Extract angles for all three planes
        angles[key]['Sagittal'] = [angle_data[0, idx, :n_frames]]      # X-rotation
        angles[key]['Frontal'] = [angle_data[1, idx, :n_frames]]       # Y-rotation  
        angles[key]['Transverse'] = [angle_data[2, idx, :n_frames]]    # Z-rotation
    
    return angles


def analyse_walk_movement(positions, angles):
    required_joints = ['pelvis', 'l_foot', 'r_foot']
    missing = [joint for joint in required_joints if joint not in positions]
    if missing:
        print(f"Missing joints for analysis: {missing}")
        print(f"Available joints: {list(positions.keys())}")
        return None
        
    pelvis = positions['pelvis']
    l_foot = positions['l_foot']
    r_foot = positions['r_foot']
    
    frames = len(pelvis)
    print(f"Analyzing {frames} frames of gait data")
    print("="*60)

    # KNEE ANALYSIS
    l_knee_sag = angles['left_knee']['Sagittal']
    r_knee_sag = angles['right_knee']['Sagittal']
    l_knee_front = angles['left_knee']['Frontal']
    r_knee_front = angles['right_knee']['Frontal']

    print("KNEE ANGLES (Sagittal Plane - Flexion/Extension):")
    print(f"   Left  - Max: {np.max(l_knee_sag):6.1f}° | Min: {np.min(l_knee_sag):6.1f}° | ROM: {np.max(l_knee_sag) - np.min(l_knee_sag):5.1f}°")
    print(f"   Right - Max: {np.max(r_knee_sag):6.1f}° | Min: {np.min(r_knee_sag):6.1f}° | ROM: {np.max(r_knee_sag) - np.min(r_knee_sag):5.1f}°")
    
    print("KNEE ANGLES (Frontal Plane - Varus/Valgus):")
    print(f"   Left ROM:  {np.max(l_knee_front) - np.min(l_knee_front):5.1f}°")
    print(f"   Right ROM: {np.max(r_knee_front) - np.min(r_knee_front):5.1f}°")

    # HIP ANALYSIS
    l_hip_sag = angles['left_hip']['Sagittal']
    r_hip_sag = angles['right_hip']['Sagittal']
    l_hip_front = angles['left_hip']['Frontal']
    r_hip_front = angles['right_hip']['Frontal']
    
    print("\nHIP ANGLES (Sagittal Plane - Flexion/Extension):")
    print(f"   Left  - Max: {np.max(l_hip_sag):6.1f}° | Min: {np.min(l_hip_sag):6.1f}° | ROM: {np.max(l_hip_sag) - np.min(l_hip_sag):5.1f}°")
    print(f"   Right - Max: {np.max(r_hip_sag):6.1f}° | Min: {np.min(r_hip_sag):6.1f}° | ROM: {np.max(r_hip_sag) - np.min(r_hip_sag):5.1f}°")
    
    print("HIP ANGLES (Frontal Plane - Abduction/Adduction):")
    print(f"   Left ROM:  {np.max(l_hip_front) - np.min(l_hip_front):5.1f}°")
    print(f"   Right ROM: {np.max(r_hip_front) - np.min(r_hip_front):5.1f}°")
    
    # FOOT PROGRESSION
    l_fp = angles['left_fp']['Transverse']
    r_fp = angles['right_fp']['Transverse']
    
    print("\nFOOT PROGRESSION (Transverse Plane - Internal/External Rotation):")
    print(f"   Left  - Mean: {np.mean(l_fp):6.1f}° | Std: {np.std(l_fp):5.1f}°")
    print(f"   Right - Mean: {np.mean(r_fp):6.1f}° | Std: {np.std(r_fp):5.1f}°")
    
    # SYMMETRY ANALYSIS
    print("\nSYMMETRY ANALYSIS:")
    knee_rom_l = np.max(l_knee_sag) - np.min(l_knee_sag)
    knee_rom_r = np.max(r_knee_sag) - np.min(r_knee_sag)
    knee_symmetry = min(knee_rom_l, knee_rom_r) / max(knee_rom_l, knee_rom_r)
    
    hip_rom_l = np.max(l_hip_sag) - np.min(l_hip_sag)
    hip_rom_r = np.max(r_hip_sag) - np.min(r_hip_sag)
    hip_symmetry = min(hip_rom_l, hip_rom_r) / max(hip_rom_l, hip_rom_r)
    
    print(f"   Knee ROM Symmetry: {knee_symmetry:.3f} (1.0 = perfect)")
    print(f"   Hip ROM Symmetry:  {hip_symmetry:.3f} (1.0 = perfect)")
    
    # CLINICAL ASSESSMENT
    print("\nCLINICAL ASSESSMENT:")
    warnings = []
    
    if np.min(l_knee_sag) < -5 or np.min(r_knee_sag) < -5:
        warnings.append("Knee hyperextension detected")
    if np.max(l_knee_sag) > 70 or np.max(r_knee_sag) > 70:
        warnings.append("Excessive knee flexion detected")
    if knee_rom_l < 30 or knee_rom_r < 30:
        warnings.append("Limited knee ROM detected")
    if abs(np.mean(l_fp)) > 15 or abs(np.mean(r_fp)) > 15:
        warnings.append("Abnormal foot progression detected")
    if knee_symmetry < 0.8 or hip_symmetry < 0.8:
        warnings.append("Significant asymmetry detected")
    
    if warnings:
        for warning in warnings:
            print(f"   WARNING: {warning}")
    else:
        print("   No major gait deviations detected")
    
    # BASIC SPATIAL METRICS
    print(f"\nSPATIAL METRICS:")
    pelvis_forward = pelvis[:, 1]  # Y-axis (forward direction)
    total_distance = abs(pelvis_forward[-1] - pelvis_forward[0])
    avg_speed = total_distance / frames
    
    print(f"   Total Distance: {total_distance:7.1f} mm ({total_distance/1000:.2f} m)")
    print(f"   Average Speed:  {avg_speed:7.2f} mm/frame")
    
    # Step width
    step_widths = []
    for frame in range(frames):
        width = abs(l_foot[frame, 0] - r_foot[frame, 0])
        step_widths.append(width)
    
    avg_step_width = np.mean(step_widths)
    print(f"   Average Step Width: {avg_step_width:6.1f} mm")
    
    print("="*60)
    
    # RETURN RESULTS
    return {
        'knee_symmetry': knee_symmetry,
        'hip_symmetry': hip_symmetry,
        'total_distance': total_distance,
        'avg_speed': avg_speed,
        'avg_step_width': avg_step_width,
        'warnings': warnings
    }

# EXECUTION CODE (OUTSIDE THE FUNCTION)
print("Processing gait data...")

# Run the analysis
results = analyse_walk_movement(positions, angles)

# Print summary
if results:
    print(f"\nSUMMARY:")
    print(f"Analysis completed successfully")
    print(f"Knee symmetry: {results['knee_symmetry']:.3f}")
    print(f"Hip symmetry: {results['hip_symmetry']:.3f}")
    print(f"Jogging speed: {results['avg_speed']:.2f} mm/frame")


def analyse_walk_movement(positions, angles):
    required_joints = ['pelvis', 'l_foot', 'r_foot']
    missing = [joint for joint in required_joints if joint not in positions]
    if missing:
        print(f"Missing joints for analysis: {missing}")
        print(f"Available joints: {list(positions.keys())}")
        return None
    pelvis = positions['pelvis']
    l_foot = positions['l_foot']
    r_foot = positions['r_foot']
    
    frames = len(pelvis)
    print(f"Analyzing {frames} frames of gait data")

    l_knee_sag = angles['left_knee']['Sagittal']
    r_knee_sag = angles['right_knee']['Sagittal']
    l_knee_front = angles['left_knee']['Frontal']
    r_knee_front = angles['right_knee']['Frontal']

    print("KNEE ANGLES (Sagittal Plane - Flexion/Extension):")
    print(f"   Left  - Max: {np.max(l_knee_sag):6.1f}° | Min: {np.min(l_knee_sag):6.1f}° | ROM: {np.max(l_knee_sag) - np.min(l_knee_sag):5.1f}°")
    print(f"   Right - Max: {np.max(r_knee_sag):6.1f}° | Min: {np.min(r_knee_sag):6.1f}° | ROM: {np.max(r_knee_sag) - np.min(r_knee_sag):5.1f}°")
    
    print("KNEE ANGLES (Frontal Plane - Varus/Valgus):")
    print(f"   Left ROM:  {np.max(l_knee_front) - np.min(l_knee_front):5.1f}°")
    print(f"   Right ROM: {np.max(r_knee_front) - np.min(r_knee_front):5.1f}°")

    # HIP ANALYSIS
    l_hip_sag = angles['left_hip']['Sagittal']
    r_hip_sag = angles['right_hip']['Sagittal']
    l_hip_front = angles['left_hip']['Frontal']
    r_hip_front = angles['right_hip']['Frontal']
    
    print("\nHIP ANGLES (Sagittal Plane - Flexion/Extension):")
    print(f"   Left  - Max: {np.max(l_hip_sag):6.1f}° | Min: {np.min(l_hip_sag):6.1f}° | ROM: {np.max(l_hip_sag) - np.min(l_hip_sag):5.1f}°")
    print(f"   Right - Max: {np.max(r_hip_sag):6.1f}° | Min: {np.min(r_hip_sag):6.1f}° | ROM: {np.max(r_hip_sag) - np.min(r_hip_sag):5.1f}°")
    
    print("HIP ANGLES (Frontal Plane - Abduction/Adduction):")
    print(f"   Left ROM:  {np.max(l_hip_front) - np.min(l_hip_front):5.1f}°")
    print(f"   Right ROM: {np.max(r_hip_front) - np.min(r_hip_front):5.1f}°")
    
    # FOOT PROGRESSION
    l_fp = angles['left_fp']['Transverse']
    r_fp = angles['right_fp']['Transverse']
    
    print("\nFOOT PROGRESSION (Transverse Plane - Internal/External Rotation):")
    print(f"   Left  - Mean: {np.mean(l_fp):6.1f}° | Std: {np.std(l_fp):5.1f}°")
    print(f"   Right - Mean: {np.mean(r_fp):6.1f}° | Std: {np.std(r_fp):5.1f}°")
    
    # SYMMETRY ANALYSIS
    print("\nSYMMETRY ANALYSIS:")
    knee_rom_l = np.max(l_knee_sag) - np.min(l_knee_sag)
    knee_rom_r = np.max(r_knee_sag) - np.min(r_knee_sag)
    knee_symmetry = min(knee_rom_l, knee_rom_r) / max(knee_rom_l, knee_rom_r)
    
    hip_rom_l = np.max(l_hip_sag) - np.min(l_hip_sag)
    hip_rom_r = np.max(r_hip_sag) - np.min(r_hip_sag)
    hip_symmetry = min(hip_rom_l, hip_rom_r) / max(hip_rom_l, hip_rom_r)
    
    print(f"   Knee ROM Symmetry: {knee_symmetry:.3f} (1.0 = perfect)")
    print(f"   Hip ROM Symmetry:  {hip_symmetry:.3f} (1.0 = perfect)")

    print("Processing gait data...")

# Extract matrices and positions
matrices = extract_matrices_final(myjog, labels)
positions = extract_positions_from_matrices(matrices)

# Extract angles  
angles = extract_angles(myjog, labels2)

# Run the analysis
results = analyse_walk_movement(positions, angles)

# Print summary
if results:
    print(f"\nSUMMARY:")
    print(f"Analysis completed successfully")


