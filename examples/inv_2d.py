import sympy as sp
import numpy as np
from time import time
import matplotlib.pyplot as plt
import sys
sys.path.append('../')

from density_sdsos import *

from plot_2d import plot2d_invariance, plot2d_surf

# Initialize symbolic variables
x1, x2, d = sp.symbols('x1,x2,d')

dmax = 0.06
data = {'x_vars': [x1, x2],
		'd_vars': [d],
		'vector_field': [x2**2-0.1*x1, -x1**2*x2-0.1+d], 

		'X': [(x1-(-2))*(2-x1), (x2-(-2))*(2-x2)],
		'D': [(d-(-dmax))*(dmax-d)],
		'K': [(x1-(-1.1))*(1.1-x1), (x2-(-1.1))*(1.1-x2)],

		'maxdeg' : 12, 
		'tol': 1e-5,
		'r': 0,
		'alpha': 0.05}


rho = compute_inv_mosek_not(data)

print "Density found: rho(x1,x2) = ", rho

fig1 = plt.figure(1)
plot2d_invariance(rho, data, [(-dmax,), (dmax,)], fig1.gca())

fig2 = plt.figure(2)
plot2d_surf(rho, data, fig2.gca(projection='3d'))

plt.show()

# animate([t,x1,x2], data['vector_field'], rho, 0, 1.1*data['T'])
