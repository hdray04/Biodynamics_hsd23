import ezc3d
import numpy as np 
import matplotlib as plt 
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


hop_single = ezc3d.c3d("/Users/harrietdray/Biodynamics/Harriet_c3d/HopSingle-001/pose_filt_0.c3d")
triple_hop_1 = ezc3d.c3d("/Users/harrietdray/Biodynamics/Harriet_c3d/HopTriple-001/pose_filt_0.c3d")
triple_hop_2 = ezc3d.c3d("/Users/harrietdray/Biodynamics/Harriet_c3d/HopTriple-002/pose_filt_0.c3d")

angle_data_trials = [hop_single['data']['points'], triple_hop_1['data']['points'], triple_hop_2['data']['points']]
rotation_data_trials = [hop_single['data']['rotations'], triple_hop_1['data']['rotations'], triple_hop_2['data']['rotations']]
hop_data = [hop_single, triple_hop_1, triple_hop_2]
labels = hop_data[0]['parameters']['POINT']['LABELS']['value'] # List of angle names
labels_rotation = hop_data[0]['parameters']['ROTATION']['LABELS']['value']
units = hop_data[0]['parameters']['POINT']['UNITS']['value']
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
        rotation_data = hop_data[trial_idx]['data']['rotations']  # Extract rotation data for the current trial
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




fs = 100  # Hz

def butter_low(x, fs, fc=8, order=2):
    b,a = butter(order, fc/(fs/2), btype='low')
    return filtfilt(b,a,x)

def detect_to_ic(foot_z, fs, thresh=None):
    """Return take-off and landing frame indices from vertical foot pos"""
    z = butter_low(foot_z, fs, fc=8)
    if thresh is None:
        thresh = np.percentile(z, 20) + 0.005
    stance = z < thresh
    ds = np.diff(stance.astype(int), prepend=stance[0])
    to_idx = np.where(ds == -1)[0]  # foot leaves ground
    ic_idx = np.where(ds == 1)[0]   # foot contacts ground
    if len(to_idx) and len(ic_idx):
        return to_idx[0], ic_idx[0]
    else:
        return None, None

def hop_time_and_distance(single_positions):
    """
    single_positions: dict from extract_positions_from_matrices()['trial_1']
                      e.g. single_positions['RightFoot'] -> (N,3)
    """
    foot = single_positions['RightFoot']  # adjust name if 'LeftFoot'
    foot_z = foot[:,2]
    to_i, ic_i = detect_to_ic(foot_z, fs)
    if to_i is None: 
        return None
    
    flight_time = (ic_i - to_i)/fs
    distance = foot[ic_i,0] - foot[to_i,0]  # assumes X is forward axis
    
    return {'TO_frame': to_i,
            'IC_frame': ic_i,
            'flight_time_s': flight_time,
            'distance_m': distance}