"""Run reduced model
Select a reduced subset of parameters
to vary in the model
"""

from model import fcn, dim_mat
import numpy as np
import matplotlib.pyplot as plt
import pyutil.numeric as ut
from numpy.linalg import norm
from scipy.linalg import svd, qr
from copy import copy

## Setup
# --------------------------------------------------
n_samp = int(1e2)
# par = range(7)                  # Parameter indices to include, ascending
par = [0,1,3,4,5,6]             # Skip A_0
m   = len(par)
tol = 1e-3                      # Tolerance for matrix_rank
u   = np.array([1,1,-2])        # Units for thrust

# Make nominal values global, len=7
P_0 = 101e3                     # Freestream pressure, Pa
P_e = P_0 * 10                  # Exit pressure, Pa
A_0 = 1.5                       # Capture area, m^2
A_e = 1.                        # Nozzle exit area, m^2
U_0 = 100.                      # Freestream velocity, m/s
U_e = U_0 * 2                   # Exit velocity, m/s
a_0 = 343.                      # Ambient sonic speed, m/s

X_nom = np.array([P_0,P_e,A_0,A_e,U_0,U_e,a_0])

## Function definitions
# --------------------------------------------------
_epsilon = np.sqrt(np.finfo(float).eps)

def grad(x,f,f0=None,h=_epsilon):
    """Computes a FD approximation to the gradient
    at a point
    """
    # If necessary, calculate f(x)
    if f0:
        pass
    else:
        f0 = f(x)

    # Ensure query point is np.array
    if type(x).__module__ == np.__name__:
        pass
    else:
        x = np.array(x)

    # Calculate gradient
    dF = np.empty(np.size(x))   # Reserve space
    E  = np.eye(np.size(x))     # Standard basis
    for i in range(np.size(x)):
        dF[i] = (f(x+E[:,i]*h)-f0)/h

    return dF

## Run script
# --------------------------------------------------
# Parameter bounds
fac = 1.15

X_l = np.array(X_nom)*fac
X_u = np.array(X_nom)/fac

# Log-Parameter bounds
logq_l = np.log(X_l); logq_u = np.log(X_u)
# Log-Objective and gradient
xpt = lambda q: np.exp(0.5*(q+1) * (logq_u-logq_l) + logq_l)
def qpt(p):
    res = copy(X_nom)
    res[par] = p
    return res

fun = lambda p: fcn(xpt(qpt(p)))

# Monte Carlo method
Q_samp = 2*np.random.random((n_samp,m)) - 1
F_samp = []
G_samp = []
for ind in range(n_samp):
    # Evaluate function at point
    # q = Q_samp[ind]
    f_val = fun(Q_samp[ind])
    g_val = grad(Q_samp[ind],fun,f0=f_val)*2/(logq_u[par]-logq_l[par])
    # Record values
    F_samp.append(f_val)
    G_samp.append(g_val)
# Construct C
G = np.array(G_samp)
C = G.T.dot(G)
# Compute Active Subspace
W,L,_ = svd(C)

# Console output
print("Computation complete!")
print("L = {}".format(L))

## Analysis
# --------------------------------------------------
D = np.array(dim_mat)[:,par]    # Take only used parameters
N = ut.null(D)

# 
as_dim = ut.as_dim(L)
W_1 = W[:,:as_dim]
dim = D.dot(W_1)
rnk = np.linalg.matrix_rank(dim,tol=tol) 

# Project out the dimensions of thrust
Qtmp,Rtmp = qr(np.concatenate((ut.col(u),dim),axis=1))
Qc = Qtmp[:,1:rnk]; Qc = Qc / Qc[0]


# Console printout
print("dim = \n{}".format(dim))
print("rank(dim) = {}".format(rnk))
