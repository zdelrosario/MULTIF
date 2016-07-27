import numpy as np
import pandas as pd
from scipy.linalg import svd, qr
import pyutil.numeric as ut
from pyutil.plotting import linspecer

import matplotlib
matplotlib.use("Qt4Agg")
import matplotlib.pyplot as plt
offset = [(0,500),(700,500),(1400,500)] # window locations

## Setup
# --------------------------------------------------
# Parameters
bs_flag = False             # Perform Bootstrap analysis?
n_resamp = int(1e3)         # Number of times to do bootstrap resample
np.set_printoptions(precision=2,linewidth=100)

# Dimensional matrix
# Column order = {T_0, K, P_inf, T_inf, E, alpha, P_0}
D = np.array([ [ 0, 1, 1, 0, 1, 0, 1],  # M
               [ 0, 2,-1, 0,-1, 2,-1],  # L
               [ 0,-3,-2, 0,-2,-1,-2],  # T
               [ 1,-1, 0, 1, 0, 0, 0] ])# Theta
N = ut.null(D)
#A = 

# Load data
df = pd.read_csv("grads.csv")
G  = df.as_matrix()[:,1:]
m  = G.shape[1]

# Compute AS
C  = G.T.dot(G)
W,L,_ = svd(C)

# Helper functions
# def bootstrap_resample(X, n=None):
#     """ Bootstrap resample an array_like
#     http://nbviewer.jupyter.org/gist/aflaxman/6871948
#     Parameters
#     ----------
#     X : array_like
#       data to resample
#     n : int, optional
#       length of resampled array, equal to len(X) if n==None
#     Results
#     -------
#     returns X_resamples
#     """
#     if n == None:
#         n = len(X)
        
#     resample_i = np.floor(np.random.rand(n)*len(X)).astype(int)
#     X_resample = X[resample_i]
#     return X_resample

## Bootstrap resample
# --------------------------------------------------
if bs_flag:
    W_all = []; L_all = []
    D_all = []
    for ind in range(n_resamp):
        # Perform the resampling
        G_r = ut.bootstrap_resample(G)
        # Compute AS
        C_r = G_r.T.dot(G_r)
        W_r,L_r,_ = svd(C_r)
        # Calculate subspace distance
        d_r = []
        for i in range(1,m):
            d_r.append(ut.subspace_distance(W[:,:i],W_r[:,:i]))

        # Record results
        W_all.append(W_r)
        L_all.append(L_r)
        D_all.append(d_r)
    # Form numpy arrays
    L_all = np.array(L_all)
    D_all = np.array(D_all)

    # Bootstrap analysis
    L_mean = np.array([L_all[:,i].mean() for i in range(L_all.shape[1])])
    L_std  = np.array([L_all[:,i].std() for i in range(L_all.shape[1])])
    D_mean = np.array([D_all[:,i].mean() for i in range(D_all.shape[1])])
    D_std  = np.array([D_all[:,i].std() for i in range(D_all.shape[1])])

    # Plot results
    # --------------------------------------------------
    colors = linspecer(2)
    sig    = 2.                     # Number of sigma to plot as interval

    # Subspace Distance
    ind = range(1,m)
    D_lo= D_mean-D_std*sig; D_lo = [max(d,1e-25) for d in D_lo]
    D_hi= D_mean+D_std*sig

    plt.figure()
    plt.plot(ind,D_mean,color=colors[0],marker='*',linestyle='-')
    plt.fill_between(ind,D_lo,D_hi,alpha=0.5,color=colors[0])
    plt.title('Subspace Distance')
    plt.yscale('log')
    plt.ylim((1e-15,max(D_mean)*10))
    plt.xlim((0,m))
    # Set plot location on screen
    manager = plt.get_current_fig_manager()
    x,y,dx,dy = manager.window.geometry().getRect()
    manager.window.setGeometry(offset[0][0],offset[0][1],dx,dy)

    # Eigenvalues
    ind = range(1,len(L_mean)+1)
    L_lo= L_mean-L_std*sig; L_lo = [max(l,1e-25) for l in L_lo]
    L_hi= L_mean+L_std*sig

    plt.figure()
    plt.plot(ind,L_mean,color=colors[1],marker='*',linestyle='-')
    plt.fill_between(ind,L_lo,L_hi,alpha=0.5,color=colors[1])
    plt.title('Eigenvalues')
    plt.yscale('log')
    plt.ylim((1e-25,max(L_mean)*10))
    plt.xlim((0,m+1))
    # Set plot location on screen
    manager = plt.get_current_fig_manager()
    x,y,dx,dy = manager.window.geometry().getRect()
    manager.window.setGeometry(offset[1][0],offset[1][1],dx,dy)

## Dimensional Analysis
# --------------------------------------------------
# Check subspace inclusion

# Find intersection of DA space and AS
W_1 = W[:,:4]
N_c = ut.comp(N)

B_1 = ut.inter(W_1,N)           # Dimensionless parameters
B_2 = ut.inter(W_1,N_c)         # Dimensional factor

d_1 = D.dot(B_2[:,0])
d_2 = D.dot(B_2[:,1])

# Show all plots
plt.show()
