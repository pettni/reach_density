'''
Module for solving the density problem

 min  || Lf rho (t,x) ||  over  D,

where Lf is the differential operator 
	Lf : rho |--> d_t rho +  d_{x_i} rho f_i(t,x),
for a vector field f.

The optimization is done over polynomial up to a 
given degree.
'''

import numpy as np
import sympy as sp
import scipy.sparse 

import sys
import time

import mosek

from polylintrans import *

from sdd import add_sdd_picos, add_sdd_mosek

def poly_to_tuple(poly, variables):
	"""
		Transform a sympy polynomial to coefficient/exponent tuple
	"""
	if isinstance(poly, list):
		return [poly_to_tuple(p, variables) for p in poly]
	poly_ex = sp.expand(poly)
	terms = sp.Poly(poly_ex, variables).terms()
	return tuple([(t0, float(t1)) for t0,t1 in terms])

def coef_to_poly(coef, num_var):
	"""
		Transform a (grlex ordered) list of coefficients to a polynomial
	"""
	ret = 0
	mon_iterator = grlex_iter( (0,) * num_var )
	for k in range(len(coef)):
		midx = next(mon_iterator)
		mon = coef[k]*sp.Symbol('t')**midx[0]
		for i in range(1,num_var):
			mon *= sp.Symbol('x%d' % i)**midx[i]
		ret += mon
	return ret

def degree(polytuple):
	""" 
	Compute the degree of a tuple representing a polynomial
	"""
	if isinstance(polytuple, list):
		if len(polytuple) == 0:
			return 0
		else:
			return max([degree(p) for p in polytuple])
	return max([sum(term[0]) for term in polytuple])

def Lf(d, vf):
	""" 
		Linear transformation representing Lf operator on a polynomial of degree d 
			Lf : p |--> D_t p + D_x p * f(t,x)
	"""
	xdim = len(vf)
	dxi = [PolyLinTrans.diff(xdim+1, d, i) for i in range(xdim+1)]
	L = dxi[0]
	for i in range(1, xdim+1):
		vf_mul = PolyLinTrans.mul_pol(xdim+1, dxi[i].d1, vf[i-1])
		L +=  vf_mul * dxi[i]
	return L

def _compute_reach_basic(data):
	deg_max = data['maxdeg_rho']
	rho_0 = poly_to_tuple(data['rho_0'], data['variables'])
	vf = poly_to_tuple(data['vector_field'], data['variables'])
	domain = poly_to_tuple(data['domain'], data['variables'])
	r = data['r']
	tol = data['tol']

	assert(deg_max % 2 == 0)

	#  Want to make 
	#    b - L rho - d_i s_i^- 
	#    b + L rho - d_i s_i^+
	#  sdsos

	num_var = len(vf) + 1

	deg_rho = deg_max + 1 - degree(vf)
	deg_sigma = deg_max - degree(domain)
	deg_sigma = deg_sigma - (deg_sigma % 2)		# make it even

	# half degrees for symmetric matrix variables
	halfdeg_max = deg_max/2
	halfdeg_sigma = deg_sigma/2	

	print "Maximal degree", deg_max
	print "deg(rho) = ", deg_rho
	print "deg(sigma) = ", deg_sigma, "\n"

	# Compute number of variables  
	# Overall equation, monomial representation
	n_max_mon = count_monomials_leq(num_var, deg_max)
	# Overall equation, square representation
	n_max_sq = count_monomials_leq(num_var, halfdeg_max) * \
				(count_monomials_leq(num_var, halfdeg_max) + 1)/2  
	
	# rho(t,x), monomial repr.
	n_rho = count_monomials_leq(num_var, deg_rho) 		   

	# sigma, square repr.
	n_sigma_sq = count_monomials_leq(num_var, halfdeg_sigma) * \
			     (count_monomials_leq(num_var, halfdeg_sigma) + 1)/2

	##########################################################
	### Construct matrices defining linear transformations ###
	##########################################################

	print "setting up matrices..."
	t_start = time.clock()

	# Linear trans b -> coef(b)
	b_b = PolyLinTrans(num_var)
	b_b[(0,)*num_var][(0,)*num_var] = 1.
	nrow, ncol, idxi, idxj, vals = b_b.to_sparse()
	A_b_b =  scipy.sparse.coo_matrix( (vals, (idxi, idxj)), shape = (n_max_mon, ncol) )

	# Linear trans coef(rho) -> coef(Lrho)
	Lrho_rho = Lf(deg_rho, vf)
	nrow, ncol, idxi, idxj, vals = Lrho_rho.to_sparse()
	A_Lfrho_rho =  scipy.sparse.coo_matrix( (vals, (idxi, idxj)), shape = (n_max_mon, ncol) )

	# Linear trans coef(s_i) -> coef(s_i d_i)
	sidi_si = [PolyLinTrans.mul_pol(num_var, deg_sigma, dom) for dom in domain]
	A_sidi_si = []
	for i in range(len(domain)):
		nrow, ncol, idxi, idxj, vals = sidi_si[i].to_sparse_matrix()
		A_sidi_si.append( scipy.sparse.coo_matrix( (vals, (idxi, idxj)), shape = (n_max_mon, ncol) ) )

	# Get identity transformation w.r.t vector representing symmetric matrix
	nrow, ncol, idxi, idxj, vals = PolyLinTrans.eye(num_var, deg_max).to_sparse_matrix()
	A_poly_K = scipy.sparse.coo_matrix((vals, (idxi, idxj)), shape = (n_max_mon, ncol))

	# Transformation rho -> rho0
	rho0_rho = PolyLinTrans.elvar(num_var, deg_rho, 0, 0)
	nrow, ncol, idxi, idxj, vals = rho0_rho.to_sparse()
	A_rho0_rho = scipy.sparse.coo_matrix( (vals, (idxi, idxj)), shape = (nrow, ncol) )
	
	b_rho0 = np.zeros(A_rho0_rho.shape[0])
	for arg in rho_0:
		b_rho0[grlex_to_index(arg[0])] = arg[1]

	print "completed in " + str(time.clock() - t_start) + "\n"

	return num_var, n_max_mon, n_max_sq, n_rho, n_sigma_sq, \
		   A_Lfrho_rho, A_b_b, A_sidi_si, A_poly_K, \
		   A_rho0_rho, b_rho0

def compute_reach_picos(data, solver = 'gurobi'):
	'''
	Solve the density problem using the parser 
	picos (http://picos.zib.de), which introduces overhead
	but supports many solvers. 
	'''
	try:
		import picos
		import cvxopt as cvx
	except Exception, e:
		raise e

	num_var, n_max_mon, n_max_sq, n_rho, n_sigma_sq, \
		   A_Lfrho_rho, A_b_b, A_sidi_si, A_poly_K, \
		   A_rho0_rho, b_rho0 = _compute_reach_basic(data)

	# make cvx spmatrices
	A_Lfrho_rho = cvx.spmatrix( A_Lfrho_rho.data, A_Lfrho_rho.row, A_Lfrho_rho.col, size = A_Lfrho_rho.shape )
	A_b_b = cvx.spmatrix( A_b_b.data, A_b_b.row, A_b_b.col, size = A_b_b.shape )
	A_sidi_si = [ cvx.spmatrix( A.data, A.row, A.col, size = A.shape ) for A in A_sidi_si] 
	A_poly_K = cvx.spmatrix( A_poly_K.data, A_poly_K.row, A_poly_K.col, size = A_poly_K.shape )
	A_rho0_rho = cvx.spmatrix( A_rho0_rho.data, A_rho0_rho.row, A_rho0_rho.col, size = A_rho0_rho.shape )
	b_rho0 = cvx.matrix(b_rho0)

	# Constraints
	#    rho(0,x) = rho0(x) 					(1)
	#    Lf rho + b - sigma_i^+ d_i  sdd		(2)
	#   -Lf rho + b - sigma_i^- d_i  sdd		(3)
	#   sigma_i^+ sdd 							(6)
	#   sigma_i^- sdd 							(7)

	print "setting up SOCP..."
	t_start = time.clock()

	# Define optimization problem
	prob = picos.Problem()

	# Add variables
	rho = prob.add_variable( 'rho', n_rho )
	b = prob.add_variable( 'b', 1 )
	K_pos = prob.add_variable( 'K_pos', n_max_sq)
	K_neg = prob.add_variable( 'K_neg', n_max_sq)
	sig_pos_i = [ prob.add_variable( 'sig_pos[%d]' %i, n_sigma_sq ) for i in range(len(A_sidi_si)) ]
	sig_neg_i = [ prob.add_variable( 'sig_neg[%d]' %i, n_sigma_sq ) for i in range(len(A_sidi_si)) ]

	# Add init constraint
	prob.add_constraint( A_rho0_rho*rho == b_rho0 )

	# Add eq constraints
	prob.add_constraint(  A_Lfrho_rho * rho + A_b_b * b 
					- picos.sum([A_sidi_si[i] * sig_pos_i[i] for i in range(len(A_sidi_si))], 'i', '[' + str(len(A_sidi_si)) + ']') 
					== A_poly_K * K_pos
				)
	prob.add_constraint( -A_Lfrho_rho * rho + A_b_b * b 
					- picos.sum([A_sidi_si[i] * sig_neg_i[i] for i in range(len(A_sidi_si))], 'i', '[' + str(len(A_sidi_si)) + ']') 
					== A_poly_K * K_neg
				)

	# Make Ks sdd
	sdd_pos = add_sdd_picos( prob, K_pos, 'K_pos')
	sdd_neg = add_sdd_picos( prob, K_neg, 'K_neg')

	# Make sigmas sdd
	for i in range(len(A_sidi_si)):
		add_sdd_picos( prob, sig_pos_i[i], 'sig_pos_' + str(i))
		add_sdd_picos( prob, sig_neg_i[i], 'sig_neg_' + str(i))

	# Minimize b
	prob.set_objective('min', b)

	print "completed in " + str(time.clock() - t_start) + "\n"

	print "optimizing..."
	t_start = time.clock()

	prob.solve(solver=solver, verbose=False)

	print "completed in " + str(time.clock() - t_start) + "\n"

	return (coef_to_poly(np.array(rho.value), num_var)[0], np.array(b.value)[0][0])

def _coo_zeros(nrow,ncol):
	'''
	Create a scipy.sparse zero matrix with 'nrow' rows and 'ncol' columns
	'''
	return scipy.sparse.coo_matrix( (nrow,ncol) )

def compute_reach_mosek(data):
	'''
	Solve the density problem using mosek
	(https://www.mosek.com).
	'''

	try:
		import mosek
	except Exception, e:
		raise e
	
	num_var, n_max_mon, n_max_sq, n_rho, n_sigma_sq, \
		   A_Lfrho_rho, A_b_b, A_sidi_si, A_poly_K, \
		   A_rho0_rho, b_rho0 = _compute_reach_basic(data)

	A_sd_s_stacked = scipy.sparse.bmat([ A_sidi_si ])

	# Constraints
	#    rho(0,x) = rho0(x) 					(1)
	#    Lf rho + b - sigma_i^+ d_i = K^+		(2)
	#   -Lf rho + b - sigma_i^- d_i = K^-		(3)
	#   sigma_i^+ sdd 							(6)
	#   sigma_i^- sdd 							(7)

	# Variable vector
	# 
	#  [ rho  b  s_1^+ ... s_D^+  s_1^- ... s_D^-  K^+ K^- ]
	#
	#  with dimensions
	#
	#   [ n_rho 1  n_sigma_sq ... n_sigma_sq n_sigma_sq ... n_sigma_sq n_max_sq n_max_sq ]

	print "setting up SOCP..."
	t_start = time.clock()

	numvar = n_rho + 1 + 2 * len(A_sidi_si) * n_sigma_sq + 2 * n_max_sq

	# Constraints
	# 
	#   A_rho0_rho * rho = b_rho0  						
	# 	A_Lfrho_rho * rho + b - sigma_i^+ d_i - A_poly_K * K^+ = 0
	#  -A_Lfrho_rho * rho + b - sigma_i^- d_i - A_poly_K * K^- = 0
	# 	
	#  K^+ sdd
	#  K^- sdd
	#  sigma_i^+ sdd
	#  sigma_i^- sdd

	n_con_init = A_rho0_rho.shape[0]
	n_con_K = n_max_mon
	numcon = n_con_init + 2 * n_con_K

	# init constr
	Aeq1 = scipy.sparse.bmat([[A_rho0_rho, _coo_zeros(n_con_init, numvar - n_rho)]], format = 'coo')
	beq1 = b_rho0

	# 'K' constr
	Aeq2 = scipy.sparse.bmat( [[ A_Lfrho_rho, A_b_b, -A_sd_s_stacked, None,           -A_poly_K, None      ],
							  [ -A_Lfrho_rho, A_b_b, None,            -A_sd_s_stacked, None,     -A_poly_K ]],
							  format = 'coo' )
	beq2 = np.zeros(2 * n_con_K)

	# put them together
	Aeq = scipy.sparse.bmat([[ Aeq1 ], [Aeq2] ], format = 'coo')
	beq = np.hstack([beq1, beq2])

	# Set up mosek environment
	env = mosek.Env() 
	task = env.Task(0,0)

	task.appendvars(numvar) 
	task.appendcons(numcon)

	# Make all vars unbounded
	task.putvarboundslice(0, numvar, [mosek.boundkey.fr] * numvar, [0.]*numvar, [0.]*numvar )

	# Add objective function
	task.putcj(n_rho, 1.)
	task.putobjsense(mosek.objsense.minimize) 

	# Add eq constraints
	task.putaijlist( Aeq.row, Aeq.col, Aeq.data )
	task.putconboundslice(0, numcon, [mosek.boundkey.fx] * numcon, beq, beq )

	# Add sdd constraints
	add_sdd_mosek( task, n_rho + 1 + 2 * len(A_sidi_si) * n_sigma_sq, n_max_sq )		    # make K^+ sdd
	add_sdd_mosek( task, n_rho + 1 + 2 * len(A_sidi_si) * n_sigma_sq + n_max_sq, n_max_sq) # make K^- sdd

	for j in range(len(A_sidi_si)):
		add_sdd_mosek( task, n_rho + 1 + j * n_sigma_sq, n_sigma_sq )				 # make sigma_i^+ sdd
		add_sdd_mosek( task, n_rho + 1 + (j + len(A_sidi_si)) * n_sigma_sq, n_sigma_sq) # make sigma_i^- sdd
	
	print "completed in " + str(time.clock() - t_start) + "\n"

	print "converting & optimizing..."
	t_start = time.clock()

	task.optimize() 

	print "completed in " + str(time.clock() - t_start) + "\n"

	# extract solution
	opt_rho = np.zeros(n_rho)
	opt_err = np.zeros(1)
	task.getxxslice( mosek.soltype.itr, 0, n_rho, opt_rho )
	task.getxxslice( mosek.soltype.itr, n_rho, n_rho+1, opt_err )

	return ( coef_to_poly(opt_rho, num_var), opt_err[0])