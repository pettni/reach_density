import sympy as sp
import numpy as np
from time import time

import sys
sys.path.append('../')

from density_sdsos import compute_inv_mosek

from plot_2d import *

# Initialize symbolic variables
t, x1, x2 = sp.symbols('t,x1,x2')

tmax = 1
data = {'t_var': [t],
		'x_vars': [x1, x2],
		'd_vars': [],
		'maxdeg_rho' : 8, 
		'vector_field': [-x2, x1], 
		'x_domain': [(1+x1)*(1-x1), (1+x2)*(1-x2)],
		'd_domain': [],
		'T': 1,
		'tol': 1e-5,
		'r': 0}

# (rho, error) = compute_reach_picos(data, solver = 'mosek')
(rho, error) = compute_inv_mosek(data)

print "Density found: rho(t,x1,x2) = ", rho
print "Error:", error

animate([t,x1,x2], [t*x2, -x1], rho, error, 1.1*tmax)