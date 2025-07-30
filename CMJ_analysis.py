import ezc3d
import numpy as np 
import matplotlib as plt 
import matplotlib.pyplot as plt


cmj_1 = ezc3d.c3d("/Users/harrietdray/Biodynamics/Harriet_c3d/CMJ-001/pose_filt_0.c3d")
cmj_2 = ezc3d.c3d("/Users/harrietdray/Biodynamics/Harriet_c3d/CMJ-002/pose_filt_0.c3d")
cmj_3 = ezc3d.c3d("/Users/harrietdray/Biodynamics/Harriet_c3d/CMJ-003/pose_filt_0.c3d")

angle_data_trials = [cmj_1['data']['points'], cmj_2['data']['points'], cmj_3['data']['points']]
rotation_data_trials = [cmj_1['data']['rotations'], cmj_2['data']['rotations'], cmj_3['data']['rotations']]
cmj_data = [cmj_1, cmj_2, cmj_3]  # List of c3d data objects
labels = cmj_data[0]['parameters']['POINT']['LABELS']['value'] # List of angle names
labels_rotation = cmj_data[0]['parameters']['ROTATION']['LABELS']['value']
units = cmj_data[0]['parameters']['POINT']['UNITS']['value']
print("Units used:", units)

def extract_3d_angles(angle_data_trials, labels):
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

    all_angles = {}
    for trial_idx, angle_data in enumerate(angle_data_trials):
        n_frames = angle_data.shape[2]
        for joint_name, joint_idx in angle_indices.items():
            # Extract sagittal, frontal, and transverse angles (assuming 3D data is stored in the first 3 rows)
            sagittal = angle_data[0, joint_idx, :n_frames]  # X-axis (sagittal plane)
            frontal = angle_data[1, joint_idx, :n_frames]   # Y-axis (frontal plane)
            transverse = angle_data[2, joint_idx, :n_frames]  # Z-axis (transverse plane)

            # Store the 3D angles for the joint
            all_angles[joint_name] = {
                'sagittal': sagittal,
                'frontal': frontal,
                'transverse': transverse
            }

        # Store the angles for this trial
        all_angles[f'trial_{trial_idx + 1}'] = all_angles

    return all_angles


# Extract 3D angles for all trials
all_angles_3d = extract_3d_angles(angle_data_trials, labels)



def extract_matrices(rotation_data_trials, labels_rotation):# Shape: (4, 4, 19, 547)
    all_matrices = {} # Dictionary to hold matrices for each joint
    
    for trial_idx, rotation_data_ in enumerate(rotation_data_trials):
        rotation_data = cmj_data[trial_idx]['data']['rotations']  # Extract rotation data for the current trial
        matrices_dict = {}
        n_joints = rotation_data.shape[2]  # 19 joints
        n_frames = rotation_data.shape[3]  # 547 frames
    
    
        for joint_idx in range(n_joints):
            if joint_idx < len(labels_rotation):
                label = labels_rotation[joint_idx]
                joint_name = label.replace('_4X4', '')
            
            # Extract all matrices for this joint across all frames
            # Shape will be (4, 4, n_frames) -> we want (n_frames, 4, 4)
                joint_matrices = rotation_data[:, :, joint_idx, :].transpose(2, 0, 1)
            
                matrices_dict[joint_name] = joint_matrices
        
        print(f"Extracted {len(matrices_dict)} joints for trial {trial_idx + 1}")
        
        # Store the matrices for this trial
        all_matrices[f'trial_{trial_idx + 1}'] = matrices_dict

    return all_matrices
    
  

def extract_positions_from_matrices(all_matrices):
    all_positions = {}

    for trial_name, matrices in all_matrices.items():
        positions_dict = {}

        for joint_name, joint_matrices in matrices.items():
            # Extract the translation (position) component from the 4x4 matrices
            # Translation is in the last column of the 4x4 matrix
            positions = joint_matrices[:, :3, 3]  # Shape: (n_frames, 3)
            positions_dict[joint_name] = positions

        all_positions[trial_name] = positions_dict

    return all_positions


def find_initial_foot_contact(all_matrices):
    all_positions = extract_positions_from_matrices(all_matrices)

    initial_contact_frames = {}

    for trial_name, positions in all_positions.items():
        print(f"Trial: {trial_name}, Available joints: {list(positions.keys())}")
        # Assuming left ankle is used to determine foot contact
        right_foot_positions = positions['r_foot']  # Shape: (n_frames, 3)
        z_coordinates = right_foot_positions[:, 2]
        min_z = np.min(z_coordinates)
        print("Min z:", min_z)
        initial_contact_frame = np.argmin(z_coordinates)

        # Find the first frame where the z-coordinate is approximately 0 (or a threshold

        initial_contact_frames[trial_name] = initial_contact_frame

    return initial_contact_frames

all_matrices = extract_matrices(rotation_data_trials, labels_rotation)

#### Plotting the knee angles at initial contact for each trial
####Knee valgus/varus('-'move outwards) should be within -5 to 10 degrees
initial_contact_frames = find_initial_foot_contact(all_matrices)
for trial_name, frame in initial_contact_frames.items():
    if trial_name == 'trial_1':  # Only process trial 1
        print(f"{trial_name}: Initial foot contact at frame {frame}")

        left_knee_flexion = -all_angles_3d[trial_name]['left_knee']['sagittal'][frame]
        left_knee_valgus = all_angles_3d[trial_name]['left_knee']['frontal'][frame]
        left_knee_rotation = all_angles_3d[trial_name]['left_knee']['transverse'][frame]
        right_knee_flexion = all_angles_3d[trial_name]['right_knee']['sagittal'][frame]
        right_knee_valgus = all_angles_3d[trial_name]['right_knee']['frontal'][frame]
        right_knee_rotation = all_angles_3d[trial_name]['right_knee']['transverse'][frame]

        print("Knee angles at initial contact frame for trial 1:")
        print(f"Left Knee Flexion = {left_knee_flexion}, Right Knee Flexion = {right_knee_flexion}")
        print(f"Left Knee Valgus = {left_knee_valgus}, Right Knee Valgus = {right_knee_valgus}")
        print(f"Left Knee Rotation = {left_knee_rotation}, Right Knee Rotation = {right_knee_rotation}")

    if trial_name == 'trial_2':
        print(f"{trial_name}: Initial foot contact at frame {frame}")

        left_knee_flexion = -all_angles_3d[trial_name]['left_knee']['sagittal'][frame]
        left_knee_valgus = all_angles_3d[trial_name]['left_knee']['frontal'][frame]
        left_knee_rotation = all_angles_3d[trial_name]['left_knee']['transverse'][frame]
        right_knee_flexion = all_angles_3d[trial_name]['right_knee']['sagittal'][frame]
        right_knee_valgus = all_angles_3d[trial_name]['right_knee']['frontal'][frame]
        right_knee_rotation = all_angles_3d[trial_name]['right_knee']['transverse'][frame]

        print("Knee angles at initial contact frame for trial 2:")
        print(f"Left Knee Flexion = {left_knee_flexion}, Right Knee Flexion = {right_knee_flexion}")
        print(f"Left Knee Valgus = {left_knee_valgus}, Right Knee Valgus = {right_knee_valgus}")
        print(f"Left Knee Rotation = {left_knee_rotation}, Right Knee Rotation = {right_knee_rotation}")
    
    if trial_name == 'trial_3': 
        print(f"{trial_name}: Initial foot contact at frame {frame}")

        left_knee_flexion = -all_angles_3d[trial_name]['left_knee']['sagittal'][frame]
        left_knee_valgus = all_angles_3d[trial_name]['left_knee']['frontal'][frame]
        left_knee_rotation = all_angles_3d[trial_name]['left_knee']['transverse'][frame]
        right_knee_flexion = all_angles_3d[trial_name]['right_knee']['sagittal'][frame]
        right_knee_valgus = all_angles_3d[trial_name]['right_knee']['frontal'][frame]
        right_knee_rotation = all_angles_3d[trial_name]['right_knee']['transverse'][frame]

        print("Knee angles at initial contact frame for trial 3:")
        print(f"Left Knee Flexion = {left_knee_flexion}, Right Knee Flexion = {right_knee_flexion}")
        print(f"Left Knee Valgus = {left_knee_valgus}, Right Knee Valgus = {right_knee_valgus}")
        print(f"Left Knee Rotation = {left_knee_rotation}, Right Knee Rotation = {right_knee_rotation}")


