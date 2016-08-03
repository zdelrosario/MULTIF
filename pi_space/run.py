import multif
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.linalg import norm
from scipy.linalg import svd

## Setup
# --------------------------------------------------
n_samp = int(1e2)

# Make nominal values global, len=28
X_nom = np.array([2.124000000000000e-01,# Spline control
                  2.269000000000000e-01,
                  2.734000000000000e-01,
                  3.218000000000000e-01,
                  3.230000000000000e-01,
                  3.343000000000000e-01,
                  3.474000000000000e-01,
                  4.392000000000000e-01,
                  4.828000000000000e-01,
                  5.673000000000000e-01,
                  6.700000000000000e-01,
                  3.238000000000000e-01,
                  2.981000000000000e-01,
                  2.817000000000000e-01,
                  2.787000000000000e-01,
                  2.797000000000000e-01,
                  2.804000000000000e-01,
                  3.36000000000000e-01,
                  2.978000000000000e-01,
                  3.049000000000000e-01,
                  3.048000000000000e-01,
                  945,                  # T_0
                  8.96,                 # K
                  60000,                # P_inf
                  262,                  # T_inf
                  8.2e10,               # E
                  2.0e-6,               # alpha
                  240000])              # P_0

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
    # Dimension of problem
    m = 7  # Inlet flow conditions

    # Fixed spline parameters
    x_geo = X_nom[:21]

    # Parameter bounds
    fac = 1.15
    T0_l  = X_nom[21] / fac; T0_u  = X_nom[21] * fac
    K_l   = X_nom[22] / fac; K_u   = X_nom[22] * fac
    Pinf_l= X_nom[23] / fac; Pinf_u= X_nom[23] * fac
    Tinf_l= X_nom[24] / fac; Tinf_u= X_nom[24] * fac
    E_l   = X_nom[25] / fac; E_u   = X_nom[25] * fac
    alp_l = X_nom[26] / fac; alp_u = X_nom[26] * fac
    P0_l  = X_nom[27] / fac; P0_u  = X_nom[27] * fac

    X_l = np.array([T0_l,K_l,Pinf_l,Tinf_l,E_l,alp_l,P0_l])
    X_u = np.array([T0_u,K_u,Pinf_u,Tinf_u,E_u,alp_u,P0_u])

    # Log-Parameter bounds
    logq_l = np.log(X_l); logq_u = np.log(X_u)
    # Log-Objective and gradient
    xpt = lambda q: np.exp(0.5*(q+1) * (logq_u-logq_l) + logq_l)
    xfn = lambda q: np.concatenate((x_geo,xpt(q)))
    fun = lambda q: run_nozzle(xfn(q))

    # Monte Carlo method
    Q_samp = 2*np.random.random((n_samp,m)) - 1
    F_samp = []
    G_samp = []
    for ind in range(n_samp):
        # Evaluate function at point
        # q = Q_samp[ind]
        f_val = fun(Q_samp[ind])
        g_val = grad(Q_samp[ind],fun,f0=f_val)*2/(logq_u-logq_l)
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
    data.to_csv('grads.csv')
