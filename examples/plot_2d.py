import numpy as np
import sympy as sp

from sympy.utilities.lambdify import lambdify
import scipy.integrate as integrate

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def plot2d_reach(variables, vf, rho, error, tfinal):

	def deriv(z,t):
		return [vf_fcn(t,z[0],z[1])[0], vf_fcn(t,z[0],z[1])[1]]

	# convert symbolic expressions to lambdas
	vf_fcn = lambdify(variables, vf)
	rho_fcn = lambdify(variables, rho)
	rho_fcn_upper = lambdify(variables, rho+error*variables[0])
	rho_fcn_lower = lambdify(variables, rho-error*variables[0])

	# compute plot data
	X1, X2 = np.mgrid[-1.5:1.5:100j, -1.5:1.5:100j]
	dens_0 = rho_fcn(0, X1, X2)
	dens = rho_fcn(tfinal, X1, X2)
	dens_neg = rho_fcn_lower(tfinal, X1, X2)
	dens_pos = rho_fcn_upper(tfinal, X1, X2)

	# init plot
	fig = plt.figure()
	ax = fig.gca()

	# Plot densities
	cset = ax.contourf(X1, X2, dens_0, colors='k', levels=[0., 1000], alpha=0.3)
	cset = ax.contour(X1, X2, dens, colors='k', linewidths=[5.0], levels=[0.])
	cset = ax.contourf(X1, X2, dens_neg, colors='r', levels=[0., 1000], alpha=0.3)
	cset = ax.contourf(X1, X2, dens_pos, colors='b', levels=[0., 1000], alpha=0.3)

	# Plot actual solutions
	tvec = np.linspace(0, tfinal, 1000)
	for theta in np.linspace(0, 2*np.pi, 20):
		zinit = [np.sqrt(0.5)*np.cos(theta), np.sqrt(0.5)*np.sin(theta)]
		z = integrate.odeint(deriv, zinit, tvec)
		x1, x2  = z.T
		plt.plot(x1[-1], x2[-1], 'ro')

	plt.show()

def plot2d_invariance(rho, data, dvec, ax):
	
	xvars = data['x_vars']
	try:
		duvars = data['d_vars']
	except Exception, e:
		duvars = data['u_vars']

	rho_fcn = sp.lambdify(xvars, rho)

	X1, X2 = np.mgrid[-2:2:500j, -2:2:500j]

	def plotdata_semi(list):
		l0 = (sp.lambdify(xvars, list[0])(X1, X2) > 0)
		for i in range(1, len(list)):
			l0 = l0 * (sp.lambdify(xvars, list[i])(X1, X2) > 0)
		return l0

	ax.contourf(X1, X2, plotdata_semi(data['K']), levels = [0.5, np.inf], alpha = 0.5, colors='green')
	ax.contour(X1, X2, rho_fcn(X1, X2), levels = [0], colors=['blue'] )

	X1, X2 = np.mgrid[-2:2:20j, -2:2:20j]

	vf1 = sp.lambdify(xvars + duvars, data['vector_field'][0])
	vf2 = sp.lambdify(xvars + duvars, data['vector_field'][1])

	for dval in dvec:
		U1 = vf1(X1, X2, *dval)
		V1 = vf2(X1, X2, *dval)

		U1_n, V1_n = U1/np.sqrt(U1**2+V1**2), V1/np.sqrt(U1**2+V1**2)

		ax.quiver(X1, X2,  U1_n, V1_n, color='black')

def plot2d_surf(rho, data, ax):
	
	rho_fcn = sp.lambdify(data['x_vars'], rho)
	X1, X2 = np.mgrid[-2:2:50j, -2:2:50j]
	Z = rho_fcn(X1, X2)
	Z[Z<-5] = -5

 	ax.plot_surface(X1, X2, Z, rstride=1, cstride=1, cmap=cm.coolwarm)


def animate(variables, vf, rho, error, tmax):

	tvec = np.linspace(0.01,tmax,100)
	ymin = xmin = -3
	ymax = xmax = 3

	# convert symbolic expressions to lambdas
	vf_fcn0 = lambdify(variables, vf[0])
	vf_fcn1 = lambdify(variables, vf[1])
	rho_fcn = lambdify(variables, rho)
	rho_fcn_upper = lambdify(variables, rho+error*variables[0])
	rho_fcn_lower = lambdify(variables, rho-error*variables[0])

	def deriv(z,t):
		return [vf_fcn0(t,z[0],z[1]), vf_fcn1(t,z[0],z[1])]

	# pre-compute plot data
	X1, X2 = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
	dens_0 = rho_fcn(0, X1, X2)
	dens = [rho_fcn(t, X1, X2) for t in tvec]
	dens_neg = [rho_fcn_lower(t, X1, X2) for t in tvec]
	dens_pos = [rho_fcn_upper(t, X1, X2) for t in tvec]

	# Plot actual solutions
	fin = []
	for t in tvec:
		tvals = np.linspace(0, t, 1000)
		zinit = [[np.sqrt(0.5)*np.cos(theta), np.sqrt(0.5)*np.sin(theta)] for theta in np.linspace(0, 2*np.pi, 10)]
		z = np.array([integrate.odeint(deriv, z0, tvals) for z0 in zinit])
		fin.append([z[:,-1,0], z[:,-1,1]])
	fin = np.array(fin)

	# init plot
	fig = plt.figure()
	ax = fig.gca()
	ax.set_xlim(xmin,xmax)
	ax.set_xlim(ymin,ymax)

	ani = animation.FuncAnimation(fig, update, len(tvec), fargs=(ax, tvec, X1, X2, dens, dens_neg, dens_pos, fin), interval=10, blit=0)
	ani.save('anim_output.mp4', fps=15)
	plt.show()

def update(i, ax, tvec, X1, X2, dens, dens_neg, dens_pos, fin):
	ax.cla()
	ax.text(2.2,-2.8,'t=' + '%.2f' % tvec[i]  )  
	cset = ax.contour(X1, X2, dens[i], colors='k', linewidths=[3.0], levels=[0.])
	cset_neg = ax.contourf(X1, X2, dens_neg[i], colors='r', levels=[0., 1000], alpha=0.3)
	cset_pos = ax.contourf(X1, X2, dens_pos[i], colors='b', levels=[0., 1000], alpha=0.3)
	real_sol = plt.plot(fin[i,0,:], fin[i,1,:], 'ro')
	return cset, cset_neg, cset_pos, real_sol
