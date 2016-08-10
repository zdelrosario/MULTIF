"""
Create a nozzle and plot it

Rick Fenrich 8/10/16
Modified: Zach del Rosario 8/10/16 
"""

import numpy as np
import matplotlib.pyplot as plt

import multif

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

# Define a nozzle
def def_nozzle(w):
    """Takes full parameter input and 
    returns the nozzle definition
    """
    # Write the input file
    filename = "input.cfg"          # The (fixed) .cfg file
    dvname   = "inputdv_all.in"     # Write variables here
    flevel   = 0                    # Fix low fidelity

    with open(dvname, 'w') as f:
        for val in w:
            f.write("{}\n".format(val))

    # Define the nozzle
    nozzle = multif.nozzle.NozzleSetup( filename, flevel );
    return nozzle

# Choose parameter values
w = np.copy(X_nom)
w[0] = 0.2750
w[1] = 0.3060

# Define the nozzles
noz_base = def_nozzle(X_nom)
noz_comp = def_nozzle(w)

# Plot the nozzles
x1 = np.linspace(0,np.max(noz_base.wall.geometry.coefs[0,:]),1000)
y1 = noz_base.wall.geometry.radius(x1)

x2 = np.linspace(0,np.max(noz_comp.wall.geometry.coefs[0,:]),1000)
y2 = noz_comp.wall.geometry.radius(x2)


plt.plot(x1,y1,'k--')
plt.plot(x2,y2,'r')
plt.show()
