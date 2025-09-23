import numpy as np
from scipy.stats import wilcoxon

# Example arrays
Markers = np.array([218.09,230.18])
Theia = np.array([249.2,196.5])

# Wilcoxon signed-rank test
stat, p = wilcoxon(Markers, Theia)
print(f"Statistic={stat}, p-value={p:.5f}")