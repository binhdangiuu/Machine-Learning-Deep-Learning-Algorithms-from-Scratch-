"""
K-Nearest Neighbors implemented from scratch using only NumPy.

STATUS: TODO - not yet implemented.

Idea: for each query point, compute distance to all training points,
take the k closest, and predict by majority vote (classification)
or average (regression). No training phase - all computation happens at predict time.
"""

import numpy as np

# TODO: implement KNN class with vectorized distance computation
