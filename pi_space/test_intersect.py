import numpy as np
import pyutil.numeric as ut

# Set the random state
np.random.seed(0)

# Build the random matrix
D = np.random.random((3,5))
N = ut.null(D)
Nc= ut.comp(N)

# Build fake DA subspace
A = np.concatenate((ut.col(Nc[:,0]),N),axis=1)

# Test intersect
A_N = ut.inter(A,N)
A_Nc= ut.inter(A,Nc)
