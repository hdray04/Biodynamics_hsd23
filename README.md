# Biodynamics Motion Analysis (hsd23)

This repository analyzes human movement (gait, jumps, hops, squats) from Theia markerless motion capture C3D files using `ezc3d`. It extracts joint angles and joint positions from 4×4 transforms, detects gait events, computes jump/hop metrics, estimates whole‑body COM, and visualizes motion (interactive and GIF).

## Purpose
- Process Theia C3D outputs to obtain joint angles and positions.
- Detect gait events (heel strike, toe‑off) and normalize cycles.
- Compute jump/hop metrics (flight time, distance, height).
- Estimate whole‑body center of mass and external forces.
- Visualize skeleton motion over time.

## Key Scripts
- `Gait cycle.py`
  - Loads walk/jog trials; extracts angles and per‑joint positions from rotation matrices.
  - Robust heel‑strike and toe‑off detection with adaptive/fallback detectors.
  - Plots events on foot Z, joint angles over time, one‑cycle overlays (left vs right), simple symmetry metrics.
  - Assumes 100 Hz sampling; positions in mm.
- `COM.py`
  - Loads a CMJ trial, reconstructs positions, computes whole‑body COM from segment mass fractions and segment CoM locations.
  - Central‑difference velocity/acceleration; external force estimate; prints sanity checks.
- `CMJ(SLVJ)_analysis.py`
  - Loads three CMJ trials; extracts 3D angles and 4×4 transforms; derives joint positions.
  - Finds initial contact from foot Z minima and computes jump height from foot vertical trajectory.
- `Hop_analysis.py`
  - Loads single and triple‑hop trials; extracts angles, transforms, positions.
  - Detects take‑off/landing via filtered foot Z thresholding; computes flight time and forward distance.
- `Squat animation mp4.py`
  - Interactive 3D skeleton viewer (slider + keyboard) for a CMJ trial; overlays ankle angles.
- `Squatanimationgif.py`
  - Non‑interactive GIF generator for squat skeleton with a moving slider annotation; overlays knee angles.

## Notebooks
- `Gait analysis for walking.ipynb`, `Angle analysis - H.ipynb`, `Positional_analysis_h.ipynb`, `Data_exploration.ipynb`, `ML.ipynb`
  - Exploratory analyses, plotting, and preliminary ML experiments related to angles/positions and event detection.

## Data & I/O
- C3D files referenced via absolute paths under `.../Harriet_c3d/...` (e.g., `Walk-001/pose_filt_0.c3d`, `CMJ-001/pose_filt_0.c3d`).
- Uses:
  - `POINT/LABELS` for angle series (e.g., `LeftKneeAngles_Theia`).
  - `ROTATION/LABELS` for 4×4 joint transforms; positions extracted from last column.
- Units: angles in degrees; positions in mm; typical sampling rate 100 Hz.

## Dependencies
- Python: `ezc3d`, `numpy`, `scipy`, `matplotlib`, `pillow` (for GIFs).

Quick install:
```bash
pip install ezc3d numpy scipy matplotlib pillow
```

## Typical Workflow
1. Read C3D → extract angles and 4×4 transforms → compute per‑joint positions.
2. Gait: detect HS/TO → select cycles → normalize → plot joint angles and symmetry overlays.
3. Jump/Hop: detect take‑off/landing from foot Z → compute flight time, distance, height.
4. COM: mass‑weighted sum of segment CoMs → velocity, acceleration, external forces.
5. Visualization: 3D skeleton across frames with metric overlays and playback.

## Known Issues / Cleanup Opportunities
- Hardcoded absolute file paths across scripts; consider config/env variables or relative paths.
- `Squatanimationgif.py` references `cmj_2` when loading labels for a squat file — should use `squat_2`.
- Angle dict assembly in `Hop_analysis.py` and `CMJ(SLVJ)_analysis.py` stores `trial_*` keys pointing to the same dict (likely unintended overwriting of per‑trial structures).
- Mixed naming/style and some redundant imports (e.g., `import matplotlib as plt` and `import matplotlib.pyplot as plt`).
- `SLDJ_analysis.py` is a stub.
- Event detection assumes specific label names (`l_foot`, `r_toes`, etc.); fragile if label sets differ.

## Suggested Next Steps
- Parameterize file paths and subject/session selection.
- Fix per‑trial angle structure and the squat labels bug.
- Add utilities for unit conversion (mm→m) and consistent sampling‑rate handling.
- Factor shared helpers (matrix/position extraction, filtering, detectors) into a small module.

