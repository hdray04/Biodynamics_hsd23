"""Combined 3D + evolution plot (with COM) — NaN-safe version for gait analysis"""

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
    """Plot timeseries robustly, skipping/cleaning NaN/Inf to avoid axis errors."""
    import numpy as np

    def clean_series(ts, ys):
        ts = np.asarray(ts, float)
        ys = np.asarray(ys, float)
        mask = np.isfinite(ts) & np.isfinite(ys)
        if not np.any(mask):
            return None, None
        ts_f = ts[mask]
        ys_f = ys[mask]
        order = np.argsort(ts_f)
        return ts_f[order], ys_f[order]

    plotted = 0
    ymins, ymaxs = [], []

    for s, lab in zip(series_list, labels):
        tt, yy = clean_series(t, s)
        if tt is None:
            continue
        ax.plot(tt, yy, label=lab, linewidth=1.5)
        plotted += 1
        ymins.append(np.nanmin(yy))
        ymaxs.append(np.nanmax(yy))

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Force (N)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.ticklabel_format(style='plain', axis='y')

    if plotted:
        y0, y1 = float(np.min(ymins)), float(np.max(ymaxs))
        if y0 == y1:
            eps = 1.0 if y0 == 0 else abs(y0) * 0.05
            ax.set_ylim(y0 - eps, y1 + eps)
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
        cursor = ax.axvline(t[0], color='red', linestyle='--', alpha=0.8, linewidth=2)
        return cursor
    else:
        ax.set_title('No valid data to plot')
        return None


if __name__ == "__main__":  # this only runs when the script is executed directly
    cutoff_freq = 6.0  # Hz
    body_mass = 54.0  # kg, assumed body mass for the whole body COM calculation
    fs = 100  # Hz

    # Update this path to the gait trial you want to analyze
    filepath = "/Users/harrietdray/Biodynamics/Harriet_c3d/walk-001/pose_filt_0.c3d"
    cmj_1, labels_rotation = utils.load_data(filepath)
    matrices_dict = utils.extract_matrices(cmj_1, labels_rotation)
    raw_positions = utils.extract_positions_from_matrices(matrices_dict)

    # Map joint names to the skeleton keys expected by threeD_plot
    def map_positions_for_skeleton(pos):
        out = {}
        def norm(s):
            return s.replace('_', '').replace('-', '').lower()
        for name, arr in pos.items():
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
            elif ('leftlowerarm' in key) or ('llarm' in key):
                out['l_larm'] = arr
            elif ('rightlowerarm' in key) or ('rlarm' in key):
                out['r_larm'] = arr
            elif ('leftupperarm' in key) or ('leftarm' in key) or ('luarm' in key):
                out['l_uarm'] = arr
            elif ('rightupperarm' in key) or ('rightarm' in key) or ('ruarm' in key):
                out['r_uarm'] = arr
            elif ('lefthand' in key) or ('lhand' in key):
                out['l_hand'] = arr
            elif ('righthand' in key) or ('rhand' in key):
                out['r_hand'] = arr
            elif 'torso' in key or 'spine' in key:
                out['torso'] = arr

        # If torso missing but head and pelvis exist, synthesize torso as midpoint
        if 'torso' not in out and 'pelvis' in out and 'head' in out:
            pel = np.asarray(out['pelvis'])
            hed = np.asarray(out['head'])
            if pel.shape == hed.shape:
                out['torso'] = 0.5 * (pel + hed)
        return out

    positions = map_positions_for_skeleton(raw_positions)

    # Optionally exempt/approximate missing distal joints so COM can compute
    # This fills certain missing keys by duplicating a nearby joint so the
    # segment exists with zero length (small bias but robust).
    def fill_missing_distals(pos):
        pairs = [
            ('l_foot', 'l_toes'),
            ('r_foot', 'r_toes'),
            ('l_larm', 'l_hand'),
            ('r_larm', 'r_hand'),
        ]
        for prox, dist in pairs:
            if prox in pos and dist not in pos:
                pos[dist] = np.asarray(pos[prox]).copy()
        return pos

    positions = fill_missing_distals(positions)
    out = com_force.compute_whole_body_com_fixed(positions, body_mass, fs, cutoff_freq=cutoff_freq)

    # Timeseries (z-component of external force)
    F_ext = out["F_ext"][:, 2]
    F_ext_smooth = out["F_ext_smooth"][:, 2]
    n = F_ext.shape[0]
    t = np.arange(n) / fs

    # COM positions for 3D
    com = out["r_com"]

    # === COMBINED PLOTTING ===

    fig = plt.figure(figsize=(16, 8))
    ax3d = plt.subplot2grid((12, 12), (0, 0), rowspan=7, colspan=4, projection='3d')
    ax_ts = plt.subplot2grid((12, 12), (0, 5), rowspan=7, colspan=7)

    # 3D scene
    artists = init_3d_artists(ax3d, positions, com)
    ax3d.set_title('3D Body Animation', fontsize=14, fontweight='bold', pad=20)

    # Debug before plotting
    print('Debug: F_ext finite any =', np.isfinite(F_ext).any(), 'len =', F_ext.shape[0])
    print('Debug: F_ext_smooth finite any =', np.isfinite(F_ext_smooth).any(), 'len =', F_ext_smooth.shape[0])

    # Time series
    cursor = plot_evolution(ax_ts, t, [F_ext, F_ext_smooth], ["Raw COM Force (Z)", "Smooth COM Force (Z)"])
    ax_ts.set_title("COM Force Over Time (Z-axis)")

    # Layout and controls
    plt.subplots_adjust(bottom=0.20, left=0.05, right=0.98, top=0.90, wspace=0.15, hspace=0.3)
    fig.text(0.5, 0.02, "Controls: Drag slider • Arrow keys: ←/→ = ±1, ↑/↓ = ±10",
             ha='center', fontsize=11,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="lightblue", alpha=0.8, edgecolor='gray'))

    def on_frame_change(idx):
        update_3d_frame(idx, artists, positions, com)
        x = t[min(max(idx, 0), n - 1)]
        if cursor is not None:
            cursor.set_xdata([x, x])
        slider.label.set_text(f'Frame: {int(idx)}/{artists["n_frames"]-1}')
        fig.canvas.draw_idle()

    slider_ax = plt.subplot2grid((12, 12), (8, 1), colspan=10)
    slider_ax.set_xticks([])
    slider_ax.set_yticks([])
    slider = Slider(slider_ax, f'Frame: 0/{artists["n_frames"]-1}', 0, artists['n_frames']-1, valinit=0, valstep=1, valfmt='%d')
    slider.poly.set_height(0.6)
    slider.vline.set_linewidth(4)
    slider.vline.set_color('darkblue')

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

    plt.show()
