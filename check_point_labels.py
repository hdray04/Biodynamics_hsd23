import sys
from typing import List

import ezc3d


def check_point_labels(paths: List[str]) -> None:
    for p in paths:
        try:
            c3d = ezc3d.c3d(p)
            params = c3d['parameters']
            point = params.get('POINT', {})
            labels = point.get('LABELS', {}).get('value', []) or []
            units = point.get('UNITS', {}).get('value', []) or []
            print(f"\n=== File: {p}")
            print(f"  Units: {units}")
            print(f"  POINT.LABELS count: {len(labels)}")
            if labels:
                print(f"  First 20 labels: {labels[:20]}")
            # basic heuristic for angle presence
            has_angles = any(isinstance(v, str) and 'angle' in v.lower() for v in labels)
            print(f"  Contains angle-like labels: {has_angles}")
        except Exception as e:
            print(f"\n=== File: {p}")
            print(f"  ERROR: {e}")


def default_paths() -> List[str]:
    return [
        "/Users/harrietdray/Pilot - Tash_c3d - Sorted/Tash_agility_left_1/Take 2025-09-12 01-49-57 PM-043/pose_filt_0.c3d",
        "/Users/harrietdray/Pilot - Tash_c3d - Sorted/Tash_agility_left_2/Take 2025-09-12 01-49-57 PM-044/pose_filt_0.c3d",
        "/Users/harrietdray/Pilot - Tash_c3d - Sorted/Tash_agility_left_3/Take 2025-09-12 01-49-57 PM-045/pose_filt_0.c3d",
    ]


if __name__ == "__main__":
    paths = sys.argv[1:] or default_paths()
    check_point_labels(paths)

