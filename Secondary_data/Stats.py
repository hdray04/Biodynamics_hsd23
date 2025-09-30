"""Small example: paired non-parametric test.

Runs a Wilcoxon signed-rank test on two small arrays. Outputs the test statistic and p-value.
"""

import numpy as np
from scipy.stats import wilcoxon

# Example arrays
Markers = np.array([43.8, 49.6, 50.2])
Theia = np.array([50.7, 57.0, 58.81])

# Wilcoxon signed-rank test
stat, p = wilcoxon(Markers, Theia)
print(f"Statistic={stat}, p-value={p:.5f}")
