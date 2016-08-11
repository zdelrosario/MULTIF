import multif
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

from numpy.linalg import norm
from scipy.linalg import svd
from math import exp

## Setup
# --------------------------------------------------
n_samp = int(1e2)

# Make nominal values global, len=9
X_nom = np.array([0.2787,       # Throat control point
                  0.3048,       # Exit control point
                  945,          # T_0
                  8.96,         # K
                  60000,        # P_inf
                  262,          # T_inf
                  8.2e10,       # E
                  2.0e-6,       # alpha
                  240000])      # P_0


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

def run_nozzle(w):
    """Takes full parameter input and evaluates
    the nozzle for thrust
    """
    # Write the input file
    filename = "input.cfg"          # The (fixed) .cfg file
    dvname   = "inputdv_all.in"     # Write variables here
    flevel   = 0                    # Fix low fidelity

    with open(dvname, 'w') as f:
        for val in w:
            f.write("{}\n".format(val))

    # Run the nozzle
    nozzle = multif.nozzle.NozzleSetup( filename, flevel );
    multif.LOWF.Run(nozzle)

    # Return results
    return nozzle.Thrust

## Run script
# --------------------------------------------------
if __name__ == "__main__":
    # Begin code timing
    t0 = time.time()
    # Dimension of problem
    m = 7+2  # Inlet flow conditions + nozzle radii

    # Parameter bounds
    fac = 1.15

    rt_l  = 0.2750; rt_u  = 0.2787
    re_l  = 0.3048; re_u  = 0.3100
    T0_l  = X_nom[2] / fac; T0_u  = X_nom[2] * fac
    K_l   = X_nom[3] / fac; K_u   = X_nom[3] * fac
    Pinf_l= X_nom[4] / fac; Pinf_u= X_nom[4] * fac
    Tinf_l= X_nom[5] / fac; Tinf_u= X_nom[5] * fac
    E_l   = X_nom[6] / fac; E_u   = X_nom[6] * fac
    alp_l = X_nom[7] / fac; alp_u = X_nom[7] * fac
    P0_l  = X_nom[8] / fac; P0_u  = X_nom[8] * fac
    
    X_l = np.array([rt_l,re_l,T0_l,K_l,Pinf_l,Tinf_l,E_l,alp_l,P0_l])
    X_u = np.array([rt_u,re_u,T0_u,K_u,Pinf_u,Tinf_u,E_u,alp_u,P0_u])

    # Log-Parameter bounds
    # logq_l = np.log(X_l); logq_u = np.log(X_u)
    # Log-Objective and gradient
    xpt = lambda q: 0.5*( (X_u-X_l)*q + (X_u+X_l) )
    fun = lambda q: run_nozzle(xpt(q))
    
    # Monte Carlo method
    Q_samp = 2*np.random.random((n_samp,m)) - 1
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

    # Complete timing
    t1 = time.time()
    # Console output
    print("Computation complete!")
    print("L = {}".format(L))
    print("t_exec = {}".format(t1-t0))
    
    # Save the gradients
    data = pd.DataFrame(G)
    data.to_csv('gradsl.csv')
