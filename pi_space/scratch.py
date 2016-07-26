import numpy as np
import pyutil.numeric as ut

# Define dimensional matrix
# Column order = {T_0, K, P_inf, T_inf, E, alpha, P_0}
D = np.array([ [ 0, 1, 1, 0, 1, 0, 1],  # M
               [ 0, 2,-1, 0,-1, 2,-1],  # L
               [ 0,-3,-2, 0,-2,-1,-2],  # T
               [ 1,-1, 0, 1, 0, 0, 0] ])# Theta

N = ut.null(D)
