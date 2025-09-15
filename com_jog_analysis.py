import argparse
import numpy as np
import matplotlib.pyplot as plt
import ezc3d

from src.utils import extract_matrices, extract_positions_from_matrices
from src.com_force import compute_whole_body_com_fixed


def infer_sampling_rate(c3d_obj, default=100.0):
    try:
        rate = c3d_obj['parameters']['POINT']['RATE']['value']
        if isinstance(rate, (list, tuple, np.ndarray)):
            return float(rate[0])
        return float(rate)
    except Exception:
        pass
    try:
        rate = c3d_obj['parameters']['ANALOG']['RATE']['value']
        if isinstance(rate, (list, tuple, np.ndarray)):
            return float(rate[0])
        return float(rate)
    except Exception:
        pass
    return float(default)


def map_to_segment_keys(positions):
    """Map Theia-like joint names to keys expected by src.utils.SEGMENTS.
    Expected: pelvis, head, l_thigh, l_shank, l_foot, l_toes,
              r_thigh, r_shank, r_foot, r_toes, l_uarm, r_uarm, l_hand, r_hand.
    """
    out = {}
    def norm(s):
        return s.replace('_', '').replace('-', '').lower()

    for name, arr in positions.items():
        key = norm(name)
        if 'pelvis' in key:
            out['pelvis'] = arr
        elif 'head' in key:
            out['head'] = arr
        elif ('leftupleg' in key) or ('leftthigh' in key) or ('lthigh' in key):
            out['l_thigh'] = arr
        elif ('rightupleg' in key) or ('rightthigh' in key) or ('rthigh' in key):
            out['r_thigh'] = arr
        elif ('leftleg' in key) or ('lshank' in key) or ('leftshank' in key):
            out['l_shank'] = arr
        elif ('rightleg' in key) or ('rshank' in key) or ('rightshank' in key):
            out['r_shank'] = arr
        elif ('leftfoot' in key) or ('lfoot' in key):
            out['l_foot'] = arr
        elif ('rightfoot' in key) or ('rfoot' in key):
            out['r_foot'] = arr
        elif ('lefttoe' in key) or ('lefttoes' in key) or ('ltoes' in key):
            out['l_toes'] = arr
        elif ('righttoe' in key) or ('righttoes' in key) or ('rtoes' in key):
            out['r_toes'] = arr
        elif ('leftupperarm' in key) or ('leftarm' in key) or ('luarm' in key):
            out['l_uarm'] = arr
        elif ('rightupperarm' in key) or ('rightarm' in key) or ('ruarm' in key):
            out['r_uarm'] = arr
        elif ('lefthand' in key) or ('lhand' in key):
            out['l_hand'] = arr
        elif ('righthand' in key) or ('rhand' in key):
            out['r_hand'] = arr
    return out


def ensure_mm(positions):
    """If positions look like meters (very small mm ranges), convert to mm."""
    pel = positions.get('pelvis')
    if pel is None or len(pel) == 0:
        return positions
    z_span = float(np.nanmax(pel[:, 2]) - np.nanmin(pel[:, 2]))
    if z_span < 3.0:  # likely meters
        return {k: (np.asarray(v) * 1000.0) for k, v in positions.items()}
    return positions


def main():
    parser = argparse.ArgumentParser(description='Compute COM and force for a jog C3D')
    parser.add_argument('--c3d', default='/Users/harrietdray/Biodynamics/Harriet_c3d/Jog-001/pose_filt_0.c3d', help='Path to C3D file')
    parser.add_argument('--mass-kg', type=float, default=54.0, help='Body mass in kg')
    parser.add_argument('--cutoff-hz', type=float, default=6.0, help='Lowpass cutoff for COM pos (Hz)')
    args = parser.parse_args()

    c3d = ezc3d.c3d(args.c3d)
    fs = infer_sampling_rate(c3d, default=100.0)

    labels_rotation = c3d['parameters']['ROTATION']['LABELS']['value']
    matrices = extract_matrices(c3d, labels_rotation)
    positions_all = extract_positions_from_matrices(matrices)

    # Map to expected segment keys and ensure units
    seg_positions = map_to_segment_keys(positions_all)
    seg_positions = ensure_mm(seg_positions)

    # Validate coverage
    required_pairs = [
        ('pelvis', 'head'),
        ('l_thigh', 'l_shank'), ('l_shank', 'l_foot'), ('l_foot', 'l_toes'),
        ('r_thigh', 'r_shank'), ('r_shank', 'r_foot'), ('r_foot', 'r_toes'),
    ]
    missing = {k for pair in required_pairs for k in pair if k not in seg_positions}
    if missing:
        print('Warning: missing segment joints for COM:', ', '.join(sorted(missing)))
        print('Available keys:', ', '.join(sorted(seg_positions.keys())))

    out = compute_whole_body_com_fixed(seg_positions, args.mass_kg, fs, cutoff_freq=args.cutoff_hz)
    r_com = out['r_com']           # mm
    v_com = out['v_com']           # mm/s
    a_com = out['a_com']           # mm/s^2
    F_ext = out['F_ext']           # N (raw)
    F_ext_smooth = out['F_ext_smooth']  # N (smoothed)

    # Summary
    com_z_range = float(np.max(r_com[:, 2]) - np.min(r_com[:, 2]))
    max_Fz = float(np.max(F_ext_smooth[:, 2]))
    bw = args.mass_kg * 9.81
    print('=== COM/Force Summary ===')
    print(f'Sampling rate: {fs:.2f} Hz')
    print(f'COM Z range:  {com_z_range:.1f} mm')
    print(f'Peak Fz:      {max_Fz:.1f} N  ({max_Fz/bw:.2f} BW)')

    # Plot vertical force over time
    t = np.arange(F_ext_smooth.shape[0]) / fs
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, F_ext_smooth[:, 2], label='Predicted vertical GRF (N)')
    ax.axhline(bw, color='gray', ls='--', lw=1, label='1 BW')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Force (N)')
    ax.set_title('COM-based force (vertical)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

