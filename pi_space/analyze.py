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
               [ 0, 1,-1, 0,-1, 2,-1],  # L
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

# Define subspaces
W_1 = W[:,:5]
N_c = ut.comp(N)
# Define projection operators
P_N = N.dot(N.T)
P_Nc= N_c.dot(N_c.T)
# Compute subspaces
B_dim = ut.inter(N_c,W_1)       # Dimensional factor
B_c,R_c = qr(np.concatenate((B_dim,W_1),axis=1))
B_c = B_c[:,1:5]

B_1 = P_N.dot(W_1);  r_1 = np.linalg.matrix_rank(B_1)  
B_2 = P_Nc.dot(W_1); r_2 = np.linalg.matrix_rank(B_2)  

B_1,_ = qr(B_1); B_1 = B_1[:,:r_1]
B_2,_ = qr(B_2); B_2 = B_2[:,:r_2]

dim_all= D.dot(W)
dim_as = D.dot(W_1)

# Console printback
print("dim_all = \n{}".format(dim_all))
print("dim_as = \n{}".format(dim_as))

print("\nThe following rank should be 1...")
print("Rank(dim_as) = {}".format(np.linalg.matrix_rank(dim_as)))


# Show all plots
plt.show()
