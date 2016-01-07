import sympy as sp
import numpy as np
from time import time
import matplotlib.pyplot as plt
import sys
sys.path.append('../')

from density_sdsos import compute_inv_mosek2

from plot_2d import animate

# Initialize symbolic variables
t, x1, x2 = sp.symbols('t,x1,x2')

tmax = 1
data = {'t_var': [t],
		'x_vars': [x1, x2],
		'd_vars': [],
		'maxdeg_rho' : 12, 
		'vector_field': [-0.3*x1 - x2, x1 - 0.3*x2], 
		'x_domain': [4 - x1**2 - x2**2],
		'K_set': [np.sqrt(3.5) - x1**2 - 5*x2**2],
		'd_domain': [],
		'T': 1,
		'tol': 1e-5,
		'r': 0,
		'alpha': 0.95}


rho = compute_inv_mosek2(data)

print "Density found: rho(t,x1,x2) = ", rho

rho0 = sp.lambdify([x1,x2], rho.subs(t,0))
rhoT = sp.lambdify([x1,x2], rho.subs(t,data['T']))

X1, X2 = np.mgrid[-2:2:1000j, -2:2:1000j]

def plot_semi(list):
	l0 = (sp.lambdify([x1,x2], list[0])(X1, X2) > 0)
	for i in range(1, len(list)):
		l0 = l0 * (sp.lambdify([x1,x2], list[i])(X1, X2) > 0)
	return l0

plt.figure(1)
# Verify invariance (8c): should be positive on K
cdiff = plt.contourf(X1,X2, rho0(X1, X2) - data['alpha']*rhoT(X1,X2), levels=[0, np.inf], alpha = 0.5)
plt.contour(X1, X2, plot_semi(data['K_set']), levels = [0.5], alpha = 0.5, colors='red')

plt.figure(2)
# Verify bounded by 1 (8e): plot should be positive on K
b1 = plt.contourf(X1,X2, 1-rho0(X1, X2), levels=[0, np.inf], alpha = 0.5)
plt.contourf(X1, X2, plot_semi(data['K_set']) , levels = [0.5, np.inf], alpha = 0.5, colors='red')

plt.figure(3)
plt.contourf(X1, X2, plot_semi(data['K_set']), levels = [0.5, np.inf], alpha = 0.5, colors='red')
plt.contourf(X1, X2, plot_semi([ -a*b for a in data['K_set'] for b in data['x_domain'] ]), levels = [0.5, np.inf], alpha = 0.5, colors='yellow')
times = np.arange(0, 1.01*data['T'], 0.1)
cmap_fcn = plt.get_cmap('autumn', len(times))
# Verify containment in K (8d): should be negative outside K
for i,time in enumerate(times):
	temp_fcn = sp.lambdify([x1,x2], rho.subs(t,time))
	c0 = plt.contour(X1,X2, temp_fcn(X1, X2), levels=[0.], colors=[cmap_fcn(i)])
plt.show()

animate([t,x1,x2], data['vector_field'], rho, 0, 1.1*data['T'])
