import numpy as np
import pandas as pd
from scipy.linalg import svd, qr
import pyutil.numeric as ut
from pyutil.plotting import linspecer

import matplotlib.pyplot as plt

# Load data
df1 = pd.read_csv("grads.csv")
G1  = df1.as_matrix()[:,1:]
m1  = G1.shape[1]

# Compute AS
C1  = G1.T.dot(G1)
W1,L1,_ = svd(C1)

# Load data
df2 = pd.read_csv("gradsl.csv")
G2  = df2.as_matrix()[:,1:]
m2  = G2.shape[1]

# Compute AS
C2  = G2.T.dot(G2)
W2,L2,_ = svd(C2)

# Compare spectra
plt.figure()
plt.plot(L1/sum(L1),'bo-',label='Pi')
plt.plot(L2/sum(L2),'ro-',label='Sigma')

plt.yscale('log')
plt.xlim((-0.5,len(L2)-0.5))
plt.ylim((1e-18,1e1))
plt.title('Eigenvalue Comparison')
plt.xlabel('Eigenvalue Index')
plt.ylabel('Eigenvalue')

plt.show()
