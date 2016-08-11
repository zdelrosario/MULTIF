from model import fcn, dim_mat
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.linalg import norm
from scipy.linalg import svd

## Setup
# --------------------------------------------------
n_samp = int(1e2)

# Make nominal values global, len=7
P_0 = 101e3                     # Freestream pressure, Pa
P_e = P_0 * 10                  # Exit pressure, Pa
A_0 = 1.5                       # Capture area, m^2
A_e = 1.                        # Nozzle exit area, m^2
U_0 = 100.                      # Freestream velocity, m/s
U_e = U_0 * 2                   # Exit velocity, m/s
a_0 = 343.                      # Ambient sonic speed, m/s

X_nom = [P_0,P_e,A_0,A_e,U_0,U_e,a_0]

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
if __name__ == "__main__":
    # Dimension of problem
    m = 7  # Inlet flow 
    
    # Parameter bounds
    fac = 1.5
    
    X_l = np.array(X_nom)*fac
    X_u = np.array(X_nom)/fac

    # Log-Parameter bounds
    # logq_l = np.log(X_l); logq_u = np.log(X_u)
    # Log-Objective and gradient
    xpt = lambda q: 0.5*( (X_u-X_l)*q + (X_u+X_l) )
    fun = lambda q: fcn(xpt(q))

    # Monte Carlo method
    Q_samp = 2*np.random.random((n_samp,m))-1
    F_samp = []
    G_samp = []
    for ind in range(n_samp):
        # Evaluate function at point
        # q = Q_samp[ind]
        f_val = fun(Q_samp[ind])
        g_val = grad(Q_samp[ind],fun,f0=f_val)*2/(X_u-X_l)
        # Record values
        F_samp.append(f_val)
        G_samp.append(g_val)
    # Construct C
    G = np.array(G_samp)
    C = G.T.dot(G)
    # Compute Active Subspace
    Q,L,_ = svd(C)

    # Console output
    print("Computation complete!")
    print("L = {}".format(L))

    # Save the gradients
    data = pd.DataFrame(G)
    data.to_csv('gradsl.csv')
