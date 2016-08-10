"""
Create a nozzle and plot it

Rick Fenrich 8/10/16
"""

import numpy as np
import matplotlib.pyplot as plt

import multif

# Provide a config filename. This config file is of the same form as found in
# the examples directory. The B-spline control points definition in the config
# file is all that is needed to plot the geometry.
filename = 'optim_inputDV.cfg'
flevel = 0 # so NozzleSetup() will work

# Setup the nozzle
nozzle = multif.nozzle.NozzleSetup( filename, flevel )

# To view the wall geometry specifically we can look at the nozzle wall 
# definition: nozzle.wall.geometry

# To view all attributes of nozzle.wall.geometry, you can use 
# nozzle.wall.geometry.__dict__

# The geometry class (found in MULTIF/multif/nozzle/geometry.py) has some 
# important methods which will calculate the geometry. For example,
# nozzle.wall.geometry.radius(x) will return the radius of the nozzle wall
# geometry at all x-coordinates in the vector x. Or, 
# nozzle.wall.geometry.area(x) will do the same with area.

x = np.linspace(0,np.max(nozzle.wall.geometry.coefs[0,:]),1000)
y = nozzle.wall.geometry.radius(x)
plt.plot(x,y)
