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
tol = 1e-3                      # Tolerance for matrix rank calculation

# Dimensional matrix
# Column order = {r_t, r_e, T_0, K, P_inf, T_inf, E, alpha, P_0}
D = np.array([ [ 0, 0, 0, 1, 1, 0, 1, 0, 1],  # M
               [ 0, 0, 0, 1,-1, 0,-1, 2,-1],  # L
               [ 1, 1, 0,-3,-2, 0,-2,-1,-2],  # T
               [ 0, 0, 1,-1, 0, 1, 0, 0, 0] ])# Theta
N = ut.null(D)
u = np.array([1,1,-2,0])        # Units for thrust
u_a = np.array([0,2,0,0])       # Units for area

# Load data
df = pd.read_csv("grads.csv")
G  = df.as_matrix()[:,1:]
m  = G.shape[1]

# Compute AS
C  = G.T.dot(G)
W,L,_ = svd(C)

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
# Check subspace inclusion -- TODO

# Select the active directions
as_dim = ut.as_dim(L)
W_1 = W[:,:as_dim]
# Compute dimensions
dim = D.dot(W_1)
rnk = np.linalg.matrix_rank(dim,tol=tol) 

# Check that thrust is in dimensions
t_res = ut.incl(ut.col(u),dim)
a_res = ut.incl(ut.col(u_a),dim)

# Project out the dimensions of thrust
Qtmp,Rtmp = qr(np.concatenate((ut.col(u),dim),axis=1))
Qc = Qtmp[:,1:rnk]; Qc = Qc / Qc[0]
Qtmp,Rtmp = qr(np.concatenate((ut.col(u_a),Qc),axis=1))
Qc2= Qtmp[:,2:rnk]

# Console printout
print("dim = \n{}".format(dim))
print("rank(dim) = {}".format(rnk))
print("t_res = {}".format(t_res))
print("a_res = {}".format(a_res))

# Show all plots
plt.show()
