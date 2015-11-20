import sympy as sp
import numpy as np
from time import time

import sys
sys.path.append('../')

from density_sdsos import compute_reach_picos, compute_reach_mosek

from plot_2d import *

# Initialize symbolic variables
t, x1, x2 = sp.symbols('t,x1,x2')

tmax = 1
data = {'variables': [t, x1, x2],   	# t must be first
		'maxdeg_rho' : 8, 
		'rho_0': 0.5 - x1**2 - x2**2,
		'vector_field': [t*x2, -x1], 
		'domain': [2-x1**2, 2-x2**2, t*(tmax-t)],
		'tol': 1e-5,
		'r': 0}

# (rho, error) = compute_reach_picos(data, solver = 'mosek')
(rho, error) = compute_reach_mosek(data)

print "Density found: rho(t,x1,x2) = ", rho
print "Error:", error

animate(data['variables'], data['vector_field'], rho, error, 1.1*tmax)