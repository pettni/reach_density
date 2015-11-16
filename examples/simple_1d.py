import sympy as sp
import numpy as np
from time import time

from sympy.utilities.lambdify import lambdify

import matplotlib.pyplot as plt
from scipy.integrate import odeint

import sys
sys.path.append('../')
from density_sdsos import compute_reach

# Initialize symbolic variables
t, x1 = sp.symbols('t,x1')

data = {'variables': [t, x1], 'maxdeg' : 6, 'vector_field': [-x1*t], 'rho_0': 1-x1**2, 'domain': [(x1+1.5) * (1.5-x1), t * ( 1 - t)], 
		'r': 1, 'tol': 1e-5}

plt.figure()

max_deg = 16
min_deg = 6
step = 2

for (i, maxdeg) in enumerate(range(min_deg,max_deg,step)):

	print "Solving for degree ", maxdeg

	data['maxdeg_rho'] = maxdeg

	t0 = time()
	(sol, error) = compute_reach(data)
	t1 = time()

	print "Solved in ", t1-t0, ", error is ", error

	sol_fcn = lambdify(data['variables'], sol)

	sol_fcn_lower = lambdify(data['variables'], sol-t*error)
	sol_fcn_upper = lambdify(data['variables'], sol+t*error)

	T, X = np.mgrid[-0:1:1000j, -1.2:1.2:1000j]
	plt.contour(X, T, sol_fcn_upper(T,X), levels=[0.], colors=[plt.get_cmap('autumn', (max_deg-min_deg)/step)(i)] )
	plt.contour(X, T, sol_fcn_lower(T,X), levels=[0.], colors=[plt.get_cmap('autumn', (max_deg-min_deg)/step)(i)] )

# plot DE solution
vf = lambdify(data['variables'], data['vector_field'])
vf_rev = lambda x,t: vf(t,x[0]) #reversed order


t_ival = np.arange(0, 1, 1./10000)
xs = odeint(vf_rev, 1, t_ival)
plt.plot(xs, t_ival, color='green', linewidth = 2)
xs = odeint(vf_rev, -1, t_ival)
plt.plot(xs, t_ival, color='green', linewidth = 2)

plt.show()