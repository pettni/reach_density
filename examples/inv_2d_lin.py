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
x1, x2, d1, d2 = sp.symbols('x1,x2,d1,d2')

tmax = 1
d1max = 0.25
d2max = 0.1
data = { # dynamics
		'x_vars': [x1, x2],
		'd_vars': [d1, d2],
		'vector_field': [-(0.2+d1)*x1-x2, x1-0.2*x2+d2], 
		
		# domains
		'X': [(x1-(-2))*(2-x1), (x2-(-2))*(2-x2)],
		'D': [(d1-(-d1max))*(d1max-d1), (d2-(-d2max))*(d2max-d2)],
		
		# specification
		'K': [(x1-(-1.5))*(1.5-x1), (x2-(-1.5))*(1.5-x2)],
		
		# parameters
		'maxdeg' : 10, 
		'tol': 1e-10, 
		'r': 0,  # currently not used
		'alpha': 0.05}


rho = compute_inv_mosek_not(data)
fig1 = plt.figure(1)
plot2d_invariance(rho, data, list(product([-d1max, d1max], [-d2max, d2max])), fig1.gca())

fig2 = plt.figure(2)
plot2d_surf(rho, data, fig2.gca(projection='3d'))

plt.show()