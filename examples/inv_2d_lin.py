import sympy as sp
import numpy as np
from time import time
import matplotlib.pyplot as plt
import sys
sys.path.append('../')

from density_sdsos import compute_inv_mosek

from plot_2d import animate

# Initialize symbolic variables
t, x1, x2, d= sp.symbols('t,x1,x2,d')

tmax = 1
dmax = 0.35
data = {'t_var': [t],
		'x_vars': [x1, x2],
		'd_vars': [d],
		'maxdeg_rho' : 10, 
		'vector_field': [-(0.2+d)*x1-x2, x1-0.2*x2], 
		'x_domain': [(x1-(-2))*(2-x1), (x2-(-2))*(2-x2)],
		'K_set': [(x1-(-1.1))*(1.1-x1), (x2-(-1.1))*(1.1-x2)],
		'd_domain': [(d-(-dmax))*(dmax-d)],
		'T': 1,
		'tol': 1e-5,
		'r': 0,
		'alpha': 0.95}


rho = compute_inv_mosek(data)

rho0 = sp.lambdify([x1,x2], rho.subs(t,0))
rhoT = sp.lambdify([x1,x2], rho.subs(t,data['T']))

X1, X2 = np.mgrid[-2:2:1000j, -2:2:1000j]

def plotdata_semi(list):
	l0 = (sp.lambdify([x1,x2], list[0])(X1, X2) > 0)
	for i in range(1, len(list)):
		l0 = l0 * (sp.lambdify([x1,x2], list[i])(X1, X2) > 0)
	return l0

plt.figure(1)
# Verify invariance (8c): should be positive on K
plt.contourf(X1,X2, rho0(X1, X2) - data['alpha']*rhoT(X1,X2), levels=[0, np.inf], alpha = 0.5)
plt.contour(X1, X2, plotdata_semi(data['K_set']), levels = [0.5], alpha = 0.5, colors='green')

plt.figure(2)
# Verify bounded by 1 (8e): plot should be positive on K
plt.contourf(X1,X2, 10-rho0(X1, X2), levels=[0, np.inf], alpha = 0.5)
plt.contourf(X1, X2, plotdata_semi(data['K_set']) , levels = [0.5, np.inf], alpha = 0.5, colors='green')

plt.figure(3)
plt.contourf(X1, X2, plotdata_semi(data['K_set']), levels = [0.5, np.inf], alpha = 0.5, colors='green')
times = np.arange(0, 1.01*data['T'], 0.1)
cmap_fcn = plt.get_cmap('autumn', len(times))
# Verify containment in K (8d): should be negative outside K
for i,time in enumerate(times):
	temp_fcn = sp.lambdify([x1,x2], rho.subs(t,time))
	c0 = plt.contour(X1,X2, temp_fcn(X1, X2), levels=[0.], colors=[cmap_fcn(i)])

X1, X2 = np.mgrid[-2:2:20j, -2:2:20j]

vf1 = sp.lambdify([x1, x2, d], data['vector_field'][0])
vf2 = sp.lambdify([x1, x2, d], data['vector_field'][1])

U1 = vf1(X1, X2, dmax)
V1 = vf2(X1, X2, dmax)

U2 = vf1(X1, X2, -dmax)
V2 = vf2(X1, X2, -dmax)

U1_n, V1_n = U1/np.sqrt(U1**2+V1**2), V1/np.sqrt(U1**2+V1**2)
U2_n, V2_n = U2/np.sqrt(U2**2+V2**2), V2/np.sqrt(U2**2+V2**2)

plt.quiver(X1, X2,  U1_n, V1_n, color='black')
plt.quiver(X1, X2,  U2_n, V2_n, color='black')

plt.figure(4)

plt.show()
