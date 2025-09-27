"""Plotting utilities for the project."""

import sys
import os

# Add the parent directory to the Python path to access src module
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

