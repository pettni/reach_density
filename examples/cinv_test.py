import sympy as sp
import numpy as np
from time import time
from itertools import product
import matplotlib.pyplot as plt
import sys
sys.path.append('../')

from density_sdsos import *

from plot_2d import plot2d_invariance, plot2d_surf

# Initialize symbolic variables
x1, x2, u = sp.symbols('x1,x2,u')

umax = 1
data = { # dynamics
		'x_vars': [x1, x2],
		'u_vars': [u],
		'vector_field': [1, u], 
		
		# domains
		'X': [(x1-(2))*(2-x1), (x2-(-2))*(2-x2)],
		'U': [(u-(-umax))*(umax-u)],
		
		# specification
		'K': [(x1-(-0))*(1-x1), (x2-(-1))*(1-x2)],
		
		# parameters
		'maxdeg' : 10, 
		'tol': 1e-10, 
		'r': 0,  # currently not used
		'alpha': 0.05}


rho = compute_reach_not(data)
fig1 = plt.figure(1)
plot2d_invariance(rho, data, [(umax,), (-umax,)], fig1.gca())

fig2 = plt.figure(2)
plot2d_surf(rho, data, fig2.gca(projection='3d'))

plt.show()