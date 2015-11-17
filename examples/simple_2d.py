import sympy as sp
import numpy as np
from time import time

import sys
sys.path.append('../')

from density_sdsos import compute_reach, compute_reach_fast

from plot_2d import *

# Initialize symbolic variables
t, x1, x2 = sp.symbols('t,x1,x2')

tmax = 1
data = {'variables': [t, x1, x2],   	# t must be first
		'maxdeg_rho' : 6, 
		'rho_0': 0.5 - x1**2 - x2**2,
		'vector_field': [t*x2, -x1], 
		'domain': [2-x1**2, 2-x2**2, t*(tmax-t)],
		'tol': 1e-5,
		'r': 0}

# degree matchup : maxdeg_rho - 1 + deg(vector_field) = maxdeg_sigma + deg(domain)

(rho, error) = compute_reach_fast(data)

print "Density found: rho(t,x1,x2) = ", rho
print "Error:", error

animate(data['variables'], data['vector_field'], rho, error, 1.1*tmax)