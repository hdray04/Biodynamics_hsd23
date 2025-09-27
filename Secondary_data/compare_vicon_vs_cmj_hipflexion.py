import os
import numpy as np
import matplotlib.pyplot as plt
import ezc3d
from scipy.signal import find_peaks

def load_vicon_left_knee_flexion(c3d_path):
    c = ezc3d.c3d(c3d_path)
    points = c['data']['points']
    labels = c['parameters']['POINT']['LABELS']['value']
    fs = float(c['parameters']['POINT']['RATE']['value'][0])
    time = np.arange(points.shape[2]) / fs
    if 'LKneeAngles' not in labels:
        raise KeyError("'LKneeAngles' not found in POINT.LABELS for Vicon file")
    lknee_idx = labels.index('LKneeAngles')
    # X channel = sagittal flexion (deg)
    lknee_x = points[0, lknee_idx, :]
    return time, lknee_x

def load_cmj_left_knee_flexion(c3d_path):
    c = ezc3d.c3d(c3d_path)
    points = c['data']['points']
    labels = c['parameters']['POINT']['LABELS']['value']
    fs2 = 100
    key = 'LeftKneeAngles_Theia'
    if key not in labels:
        raise KeyError(f"'{key}' not found in POINT.LABELS for CMJ file. Available: e.g., {labels[:10]}")
    idx = labels.index(key)
    lknee_sag = points[0, idx, :]
    time2 = np.arange(points.shape[2]) / fs2
    return time2, lknee_sag

def main():
    vicon_single_hop_path = \
        "/Users/harrietdray/Baseline/Tash_single_hop_left_2.c3d"
    cmj_c3d_path = \
        '/Users/harrietdray/Library/CloudStorage/OneDrive-ImperialCollegeLondon/ACL_data/Pilot - Tash/Pilot - Tash_c3d - Sorted/single_hop_left_2/pose_filt_0.c3d'
    v_time, v_lknee = load_vicon_left_knee_flexion(vicon_single_hop_path)
    c_time, c_lknee = load_cmj_left_knee_flexion(cmj_c3d_path)

    align_and_plot_peaks(v_time, v_lknee, c_time, c_lknee)

    plt.figure(figsize=(12, 5))
    plt.plot(v_time, v_lknee, label='Vicon Single Hop – Left Knee Flexion (deg)')
    plt.plot(c_time, c_lknee, label='Theia – Left Knee Flexion (deg)', alpha=0.85)
    plt.xlabel('Time (s)')
    plt.ylabel('Knee Flexion (deg)')
    plt.title('Left Knee Flexion – Vicon vs Theia')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_dir = 'figures'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'overlay_left_knee_flexion_vicon_vs_cmj_unaligned.png')
    plt.savefig(out_path, dpi=200)
    print(f"Saved unaligned overlay figure to {out_path}")
    try:
        plt.show()
    except Exception as e:
        print("Could not display plot window. Please open the saved image manually.")

def align_and_plot_peaks(v_time, v_lknee, c_time, c_lknee):
    v_peaks, _ = find_peaks(v_lknee)
    c_peaks, _ = find_peaks(c_lknee)
    if len(v_peaks) == 0 or len(c_peaks) == 0:
        print("No peaks found in one or both signals.")
        return

    v_peak_idx = v_peaks[np.argmax(v_lknee[v_peaks])]
    c_peak_idx = c_peaks[np.argmax(c_lknee[c_peaks])]

    v_peak_time = v_time[v_peak_idx]
    c_peak_time = c_time[c_peak_idx]
    time_shift = v_peak_time - c_peak_time

    c_time_aligned = c_time + time_shift

    plt.figure(figsize=(12, 5))
    plt.plot(v_time, v_lknee, label='Vicon Single Hop – Left Knee Flexion (deg)')
    plt.plot(c_time_aligned, c_lknee, label='Theia – Left Knee Flexion (deg, aligned)', alpha=0.85)
    plt.xlabel('Time (s)')
    plt.ylabel('Knee Flexion (deg)')
    plt.title('Overlay (Aligned Peaks): Left Knee Flexion – Vicon vs Theia')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_dir = 'figures'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'overlay_left_knee_flexion_vicon_vs_cmj_aligned.png')
    plt.savefig(out_path, dpi=200)
    print(f"Saved aligned overlay figure to {out_path}")
    try:
        plt.show()
    except Exception as e:
        print("Could not display plot window. Please open the saved image manually.")

if __name__ == '__main__':
    main()
