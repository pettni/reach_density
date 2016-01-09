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

def coef_to_poly(coef, variables):
	"""
		Transform a (grlex ordered) list of coefficients to a polynomial
	"""
	ret = 0
	mon_iterator = grlex_iter( (0,) * len(variables) )
	for k in range(len(coef)):
		midx = next(mon_iterator)
		mon = coef[k]*variables[0]**midx[0]
		for i in range(1,len(variables)):
			mon *= variables[i]**midx[i]
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
	tot_dim = len(vf[0][0][0])
	dxi = [PolyLinTrans.diff(tot_dim, d, i) for i in range(xdim+1)]
	L = dxi[0]
	for i in range(1, xdim+1):
		vf_mul = PolyLinTrans.mul_pol(tot_dim, dxi[i].d1, vf[i-1])
		L +=  vf_mul * dxi[i]
	return L

def _compute_reach_basic(data):
	deg_max = data['maxdeg_rho']
	rho_0 = poly_to_tuple(data['rho_0'], data['x_vars'])
	vf = poly_to_tuple(data['vector_field'], data['t_var'] + data['x_vars'] + data['d_vars'])
	domain = poly_to_tuple(data['domain'], data['t_var'] + data['x_vars'] + data['d_vars'])
	r = data['r']
	tol = data['tol']

	assert(deg_max % 2 == 0)

	#  Want to make 
	#    b - L rho - dom . sigma^1
	#    b + L rho - dom . sigma^2
	#  sdsos

	num_t_var = 1
	num_x_var = len(data['x_vars'])
	num_d_var = len(data['d_vars'])

	deg_rho = deg_max + 1 - degree(vf)
	deg_sigma = [deg_max - degree(dom) for dom in domain]

	# make all degrees even
	for i in range(len(domain)):
		deg_sigma[i] = deg_sigma[i] - (deg_sigma[i] % 2)

	# half degrees for symmetric matrix variables
	halfdeg_max = deg_max/2
	halfdeg_sigma = [deg/2 for deg in deg_sigma]

	print "Maximal degree", deg_max
	print "deg(rho) = ", deg_rho
	print "deg(sigma) = ", deg_sigma, "\n"

	# Compute number of variables  
	# Overall equation, monomial representation
	n_max_mon = count_monomials_leq(num_t_var + num_x_var + num_d_var, deg_max)

	# Overall equation, square representation
	n_max_sq = count_monomials_leq(num_t_var + num_x_var + num_d_var, halfdeg_max) * \
				(count_monomials_leq(num_t_var + num_x_var + num_d_var, halfdeg_max) + 1)/2  
	
	# rho(t,x), monomial repr.
	n_rho = count_monomials_leq(num_t_var + num_x_var, deg_rho) 		   

	# sigma, square repr.
	n_sigma_sq = [count_monomials_leq(num_t_var + num_x_var + num_d_var, half_deg) * \
			     (count_monomials_leq(num_t_var + num_x_var + num_d_var, half_deg) + 1)/2 \
			     for  half_deg in halfdeg_sigma]

	##########################################################
	### Construct matrices defining linear transformations ###
	##########################################################

	print "setting up matrices..."
	t_start = time.clock()

	num_var = num_t_var + num_x_var + num_d_var

	# Linear trans b -> coef(b)
	b_b = PolyLinTrans(num_var, num_var)
	b_b[(0,)*num_var][(0,)*num_var] = 1.
	nrow, ncol, idxi, idxj, vals = b_b.to_sparse()
	A_b_b =  scipy.sparse.coo_matrix( (vals, (idxi, idxj)), shape = (n_max_mon, ncol) )

	# Linear trans coef(rho) -> coef(Lrho)
	Lrho_rho = Lf(deg_rho, vf) * PolyLinTrans.eye(num_t_var+num_x_var, num_var, deg_rho)
	nrow, ncol, idxi, idxj, vals = Lrho_rho.to_sparse()
	A_Lfrho_rho =  scipy.sparse.coo_matrix( (vals, (idxi, idxj)), shape = (n_max_mon, ncol) )

	# Linear trans coef(sigma_x) -> coef(dom_x sigma_x)
	sigma_x_domsigma_x = [PolyLinTrans.mul_pol(num_var, deg_s, dom) \
							for (deg_s, dom) in zip(deg_sigma, domain) ]
	A_sigma_x_domsigma_x = []
	for i in range(len(domain)):
		nrow, ncol, idxi, idxj, vals = sigma_x_domsigma_x[i].to_sparse_matrix()
		A_sigma_x_domsigma_x.append( scipy.sparse.coo_matrix( (vals, (idxi, idxj)), shape = (n_max_mon, ncol) ) )

	# Get identity transformation w.r.t vector representing symmetric matrix
	nrow, ncol, idxi, idxj, vals = PolyLinTrans.eye(num_var, num_var, deg_max).to_sparse_matrix()
	A_poly_K = scipy.sparse.coo_matrix((vals, (idxi, idxj)), shape = (n_max_mon, ncol))

	# Transformation rho -> rho0
	rho0_rho = PolyLinTrans.elvar(num_t_var+num_x_var, deg_rho, 0, 0)
	nrow, ncol, idxi, idxj, vals = rho0_rho.to_sparse()
	A_rho0_rho = scipy.sparse.coo_matrix( (vals, (idxi, idxj)), shape = (nrow, ncol) )
	
	b_rho0 = np.zeros(A_rho0_rho.shape[0])
	for exp, coef in rho_0:
		b_rho0[grlex_to_index(exp)] = coef

	print "completed in " + str(time.clock() - t_start) + "\n"

	return num_t_var, num_x_var, num_d_var, n_max_mon, n_max_sq, n_rho, n_sigma_sq, \
		   A_Lfrho_rho, A_b_b, A_sigma_x_domsigma_x, A_poly_K, \
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

	num_t_var, num_x_var, num_d_var, n_max_mon, n_max_sq, n_rho, n_sigma_sq, \
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
	sig_pos_i = [ prob.add_variable( 'sig_pos[%d]' %i, n_sigma_sq[i] ) for i in range(len(A_sidi_si)) ]
	sig_neg_i = [ prob.add_variable( 'sig_neg[%d]' %i, n_sigma_sq[i] ) for i in range(len(A_sidi_si)) ]

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

	return (coef_to_poly(np.array(rho.value), data['t_var'] + data['x_vars'])[0], \
			np.array(b.value)[0][0])

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
	
	num_t_var, num_x_var, num_d_var, n_max_mon, n_max_sq, n_rho, n_sigma_sq, \
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

	print "setting up Mosek SOCP..."
	t_start = time.clock()

	numvar = n_rho + 1 + 2 * sum(n_sigma_sq) + 2 * n_max_sq

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
	add_sdd_mosek( task, n_rho + 1 + 2 * sum(n_sigma_sq), n_max_sq )		    # make K^+ sdd
	add_sdd_mosek( task, n_rho + 1 + 2 * sum(n_sigma_sq) + n_max_sq, n_max_sq) # make K^- sdd

	for j in range(len(A_sidi_si)):
		add_sdd_mosek( task, n_rho + 1 + sum(n_sigma_sq[:j]), n_sigma_sq[j] )				 # make sigma_i^+ sdd
		add_sdd_mosek( task, n_rho + 1 + sum(n_sigma_sq) + sum(n_sigma_sq[:j]), n_sigma_sq[j]) # make sigma_i^- sdd
	
	print "completed in " + str(time.clock() - t_start) + "\n"

	print "optimizing..."
	t_start = time.clock()

	task.optimize() 

	print "completed in " + str(time.clock() - t_start) + "\n"

	# extract solution
	opt_rho = np.zeros(n_rho)
	opt_err = np.zeros(1)
	task.getxxslice( mosek.soltype.itr, 0, n_rho, opt_rho )
	task.getxxslice( mosek.soltype.itr, n_rho, n_rho+1, opt_err )

	return ( coef_to_poly(opt_rho, data['t_var'] + data['x_vars']), opt_err[0])

def add_sdsos(T, b, domain, variables, maxdeg):
	#  Assume that Aeq coef(p) = beq describe a polynomial equality system
	#  This function adds the constraint
	# 
	#  T p(x) - b sdsos on D
	# 
	# By adding A_T coef(p) - coef(b) - s_i g_i

	g = poly_to_tuple(domain, variables)
	deg_s = [2*np.floor((maxdeg - degree(gi))/2) for gi in g]	# degree of multipliers
	halfdeg_s = [deg/2 for deg in deg_s]						# degree of half multipliers

	# same number of variables
	assert(T.n1 == len(variables))

	# even maximal degree
	assert(maxdeg % 2 == 0)

	# room for (quadratic) multipliers?
	assert(maxdeg >= degree(g) + 2)

	numcon = count_monomials_leq(len(variables), maxdeg)

	# transformation block
	_, ncol, idxi, idxj, vals = T.to_sparse()
	Aeq1 = scipy.sparse.coo_matrix( (vals, (idxi, idxj)), shape = (numcon, ncol) )

	# number of coefficients
	n_s = [count_monomials_leq(len(variables), halfdeg) * \
			(count_monomials_leq(len(variables), halfdeg) + 1)/2 for halfdeg in halfdeg_s]

	n_K = count_monomials_leq(len(variables), maxdeg/2) * \
			(count_monomials_leq(len(variables), maxdeg/2) + 1)/2

	# multiplier block
	mul_gi = [PolyLinTrans.mul_pol(len(variables), deg_i, g_i) \
							for (deg_i, g_i) in zip(deg_s, g) ]
	A_mul_gi = []
	for mul in mul_gi:
		nrow, ncol, idxi, idxj, vals = mul.to_sparse_matrix()
		assert(nrow == numcon)
		A_mul_gi.append( scipy.sparse.coo_matrix( (vals, (idxi, idxj)), shape = (numcon, ncol) ) )
	A_gi = scipy.sparse.bmat([ A_mul_gi ], format='coo')

	nrow, ncol, idxi, idxj, vals = PolyLinTrans.eye(len(variables), len(variables), maxdeg).to_sparse_matrix()
	assert(nrow == numcon)
	A_K = scipy.sparse.coo_matrix( (vals, (idxi, idxj)), shape = (numcon, ncol) )

	Aeq2 = scipy.sparse.bmat([[-A_gi, -A_K]], format='coo')
	# positions of positive variables
	pos_sdsos = [ (sum(n_s[:j]), n_s[j]) for j in range(len(n_s)) ] + [(sum(n_s), n_K)]
	
	beq = np.zeros(numcon)
	for exp, coef in poly_to_tuple(b, variables):
		beq[grlex_to_index(exp)] = coef

	# return Aeq, beq, pos of sdsos vars
	
	return Aeq1, Aeq2, beq, pos_sdsos

def compute_inv_mosek(data):
	'''
	Solve the density problem using mosek
	(https://www.mosek.com).
	'''

	try:
		import mosek
	except Exception, e:
		raise e
	
	deg_max = data['maxdeg_rho']
	vf = poly_to_tuple(data['vector_field'], data['t_var'] + data['x_vars'] + data['d_vars'])
	t_f = data['T']
	r = data['r']
	tol = data['tol']
	alpha = data['alpha']

	assert(deg_max % 2 == 0)

	t_domain = [data['t_var'][0]*(t_f-data['t_var'][0])];
	XK_set = [data['x_domain'] + [-k_dom] for k_dom in data['K_set']]
	# XK_set = [[-a * b for a in data['K_set'] for b in data['x_domain']]]

	#  Want to make 
	#    L rho - s_T^txd g_T - s_K^txd g_K - s_D^txd g_D  sdsos   (vars t,x,d)  (1)
	#    rho(0,x) - alpha * rho(t_f, x) - s_K^1x g_K	  sdsos   (vars x) 		(2)
	#  	 -rho(t,x) - s_T^tx g_T(t) - s_X\K^tx (- g_X g_K) sdsos   (vars t,x)    (3)
	#   1 - rho(0,x) - s_K^2x g_K 					      sdsos   (vars x)   	(4)

	# Number of states of different kinds
	########################################
	num_t_var = 1
	num_x_var = len(data['x_vars'])
	num_d_var = len(data['d_vars'])

	deg_rho = deg_max + 1 - degree(vf)

	print "setting up matrices..."
	t_start = time.clock()

	# eq (1)
	T_eq1 = Lf(deg_rho, vf) * PolyLinTrans.eye(num_t_var + num_x_var, num_t_var + num_x_var + num_d_var, deg_rho)
	Aeq11, Aeq12, beq1, pos_sdsos1 = add_sdsos(T_eq1, data['tol'], \
			t_domain + data['K_set'] + data['d_domain'], \
			data['t_var'] + data['x_vars'] + data['d_vars'], \
			deg_max)

	# eq (2)
	T_eq2 = PolyLinTrans.elvar(num_t_var+num_x_var, deg_rho, 0, 0) - \
			PolyLinTrans.elvar(num_t_var+num_x_var, deg_rho, 0, data['T']) * data['alpha']

	Aeq21, Aeq22, beq2, pos_sdsos2 = add_sdsos(T_eq2, data['tol'], \
			data['K_set'], \
			data['x_vars'], \
			deg_max)

	# eq (4)
	T_eq4 = -PolyLinTrans.elvar(num_t_var+num_x_var, deg_rho, 0, 0)
	Aeq41, Aeq42, beq4, pos_sdsos4 = add_sdsos(T_eq4, -1+data['tol'], \
			data['K_set'], \
			data['x_vars'], \
			deg_max)

	Aeq = scipy.sparse.bmat([[ Aeq11, Aeq12, None,  None], 
							 [ Aeq21, None,  Aeq22, None],
							 [ Aeq41, None,  None,  Aeq42]])
	beq = np.hstack([beq1, beq2, beq4])

	pos_sdsos = [(Aeq11.shape[1] + pos, length) for (pos, length) in pos_sdsos1] + \
				[(Aeq11.shape[1] + Aeq12.shape[1] + pos, length) for (pos, length) in pos_sdsos2] + \
				[(Aeq11.shape[1] + Aeq12.shape[1] + Aeq22.shape[1] + pos, length) for  (pos, length) in pos_sdsos4]
	# eq(3)
	T_eq3 = -PolyLinTrans.eye(num_t_var + num_x_var, num_t_var + num_x_var, deg_rho)
	for XK_set_i in XK_set:
		Aeq31, Aeq32, beq3, pos_sdsos3 = add_sdsos(T_eq3, data['tol'], \
				t_domain + XK_set_i, \
				data['t_var'] + data['x_vars'], \
				deg_max)
		Aeq31 = scipy.sparse.bmat([[Aeq31, _coo_zeros(Aeq31.shape[0], Aeq.shape[1] - Aeq31.shape[1])]])

		pos_sdsos += [(Aeq.shape[1] + pos, length) for (pos, length) in pos_sdsos3]

		Aeq = scipy.sparse.bmat([[Aeq, None], [Aeq31, Aeq32]])
		beq = np.hstack([beq, beq3])

	# objective
	int_trans = PolyLinTrans.integrate(num_x_var, deg_rho, range(num_x_var), [[-1,1]]*num_x_var) * PolyLinTrans.elvar(num_t_var + num_x_var, deg_rho, 0, 0)
	_, _, _, obj_idx, obj_vals = int_trans.to_sparse()

	print "completed in " + str(time.clock() - t_start) + "\n"
	print "setting up Mosek SOCP..."
	t_start = time.clock()

	# Set up mosek environment
	env = mosek.Env() 

	task = env.Task(0,0)

	numcon = Aeq.shape[0]
	numvar = Aeq.shape[1]

	task.appendvars(numvar) 
	task.appendcons(numcon)

	# Make all vars unbounded
	task.putvarboundslice(0, numvar, [mosek.boundkey.fr] * numvar, [0.]*numvar, [0.]*numvar )

	# Add objective function
	for (i,v) in zip(obj_idx, obj_vals):
		task.putcj(i, v)
	task.putobjsense(mosek.objsense.maximize) 

	# Add eq constraints
	task.putaijlist( Aeq.row, Aeq.col, Aeq.data )
	task.putconboundslice(0, numcon, [mosek.boundkey.fx] * numcon, beq, beq )

	# Add sdd constraints
	for pos, length in pos_sdsos:
		add_sdd_mosek( task, pos, length )

	print "completed in " + str(time.clock() - t_start) + "\n"

	print "optimizing..."
	t_start = time.clock()

	task.optimize() 

	solsta = task.getsolsta(mosek.soltype.itr)

	if (solsta == mosek.solsta.optimal or solsta == mosek.solsta.near_optimal): 
		print "completed in " + str(time.clock() - t_start) + "\n"
		print "status", solsta
		# extract solution
		opt_rho = np.zeros(Aeq11.shape[1])
		task.getxxslice( mosek.soltype.itr, 0, Aeq11.shape[1], opt_rho )

		return coef_to_poly(opt_rho, data['t_var'] + data['x_vars'])

	else: 
		print "Optimal solution not found"
		print solsta
		return None

def compute_inv_mosek_not(data):
	'''
	Set up and solve
     sup \int_{K} rho(x) dx
     s.t. grad rho * f(x,d) >= -alpha * rho(x)   on K x D 	(1)
          rho(x) <= 0							 on X\K     (2)
          rho(x) <= 1							 on K       (3)
    using (https://www.mosek.com).
	'''

	try:
		import mosek
	except Exception, e:
		raise e
	
	deg_max = data['maxdeg']
	assert(deg_max % 2 == 0)

	vf = poly_to_tuple(data['vector_field'], data['x_vars'] + data['d_vars'])


	XK_set = [data['X'] + [-k_dom] for k_dom in data['K']]

	# Number of states of different kinds
	########################################
	num_x_var = len(data['x_vars'])
	num_d_var = len(data['d_vars'])

	deg_rho = deg_max + 1 - degree(vf)

	print "setting up matrices..."
	t_start = time.clock()

	# eq (1)
	T_eq1 = PolyLinTrans(num_x_var + num_d_var, num_x_var + num_d_var)
	for i in range(num_x_var):
		dx = PolyLinTrans.diff(num_x_var + num_d_var, deg_rho, i)
		vf_mul = PolyLinTrans.mul_pol(num_x_var + num_d_var, dx.d1, vf[i])
		T_eq1 +=  vf_mul * dx
	T_eq1 = T_eq1 * PolyLinTrans.eye(num_x_var, num_x_var + num_d_var, deg_rho)
	
	T_eq1 += PolyLinTrans.eye(num_x_var, num_x_var + num_d_var, deg_rho) * data['alpha']

	Aeq11, Aeq12, beq1, pos_sdsos1 = add_sdsos(T_eq1, data['tol'], \
			data['K'] + data['D'], \
			data['x_vars'] + data['d_vars'], \
			deg_max)

	# eq (3)
	T_eq3 = -PolyLinTrans.eye(num_x_var, num_x_var, deg_rho)
	Aeq31, Aeq32, beq3, pos_sdsos3 = add_sdsos(T_eq3, -1+data['tol'], \
			data['K'], \
			data['x_vars'], \
			deg_max)

	Aeq = scipy.sparse.bmat([[ Aeq11, Aeq12, None], [Aeq31, None, Aeq32]], format='coo') 
	beq = np.hstack([beq1, beq3])

	pos_sdsos = [(Aeq11.shape[1] + pos, length) for (pos, length) in pos_sdsos1] + \
				[(Aeq11.shape[1] + Aeq12.shape[1] + pos, length) for (pos, length) in pos_sdsos3]

	# eq(2)
	T_eq2 = -PolyLinTrans.eye(num_x_var, num_x_var, deg_rho)
	for XK_set_i in XK_set:
		Aeq21, Aeq22, beq2, pos_sdsos2 = add_sdsos(T_eq2, data['tol'], \
				XK_set_i, \
				data['x_vars'], \
				deg_max)
		Aeq21 = scipy.sparse.bmat([[Aeq21, _coo_zeros(Aeq21.shape[0], Aeq.shape[1] - Aeq21.shape[1])]])

		pos_sdsos += [(Aeq.shape[1] + pos, length) for (pos, length) in pos_sdsos2]

		Aeq = scipy.sparse.bmat([[Aeq, None], [Aeq21, Aeq22]])
		beq = np.hstack([beq, beq2])


	# objective
	int_trans = PolyLinTrans.integrate(num_x_var, deg_rho, range(num_x_var), [[-1,1]]*num_x_var)
	_, _, _, obj_idx, obj_vals = int_trans.to_sparse()

	print "completed in " + str(time.clock() - t_start) + "\n"
	print "setting up Mosek SOCP..."
	t_start = time.clock()

	# Set up mosek environment
	env = mosek.Env() 

	task = env.Task(0,0)

	numcon = Aeq.shape[0]
	numvar = Aeq.shape[1]

	task.appendvars(numvar) 
	task.appendcons(numcon)

	# Make all vars unbounded
	task.putvarboundslice(0, numvar, [mosek.boundkey.fr] * numvar, [0.]*numvar, [0.]*numvar )

	# Add objective function
	for (i,v) in zip(obj_idx, obj_vals):
		task.putcj(i, v)
	task.putobjsense(mosek.objsense.maximize) 

	# Add eq constraints
	task.putaijlist( Aeq.row, Aeq.col, Aeq.data )
	task.putconboundslice(0, numcon, [mosek.boundkey.fx] * numcon, beq, beq )

	# Add sdd constraints
	for pos, length in pos_sdsos:
		add_sdd_mosek( task, pos, length )

	print "completed in " + str(time.clock() - t_start) + "\n"

	print "optimizing..."
	t_start = time.clock()

	task.optimize() 

	solsta = task.getsolsta(mosek.soltype.itr)

	if (solsta == mosek.solsta.optimal or solsta == mosek.solsta.near_optimal): 
		print "completed in " + str(time.clock() - t_start) + "\n"
		print "status", solsta
		# extract solution
		opt_rho = np.zeros(Aeq11.shape[1])
		task.getxxslice( mosek.soltype.itr, 0, Aeq11.shape[1], opt_rho )

		return coef_to_poly(opt_rho, data['x_vars'])

	else: 
		print "Optimal solution not found"
		print solsta
		return None

def compute_cinv_mosek_not(data):
	'''
	Set up and solve
     sup \int_{K} rho(x) dx
     s.t. grad rho * f(x,d) >= alpha * rho(x)    on K x U 	(1)
          rho(x) >= 0							 on X\K     (2)
          rho(x) <= 1							 on K       (3)
    using (https://www.mosek.com).
	'''

	try:
		import mosek
	except Exception, e:
		raise e
	
	deg_max = data['maxdeg']
	assert(deg_max % 2 == 0)

	vf = poly_to_tuple(data['vector_field'], data['x_vars'] + data['u_vars'])

	# X\K set
	XK_set = [data['X'] + [-ki] for ki in data['K']]

	# Number of states of different kinds
	########################################
	num_x_var = len(data['x_vars'])
	num_u_var = len(data['u_vars'])

	deg_rho = deg_max + 1 - degree(vf)

	print "setting up matrices..."
	t_start = time.clock()

	# eq (1)
	T_eq1 = PolyLinTrans(num_x_var + num_u_var, num_x_var + num_u_var)
	for i in range(num_x_var):
		dx = PolyLinTrans.diff(num_x_var + num_u_var, deg_rho, i)
		vf_mul = PolyLinTrans.mul_pol(num_x_var + num_u_var, dx.d1, vf[i])
		T_eq1 +=  vf_mul * dx
	T_eq1 = T_eq1 * PolyLinTrans.eye(num_x_var, num_x_var + num_u_var, deg_rho)
	
	T_eq1 -= PolyLinTrans.eye(num_x_var, num_x_var + num_u_var, deg_rho) * data['alpha']

	Aeq11, Aeq12, beq1, pos_sdsos1 = add_sdsos(T_eq1, data['tol'], \
			data['K'] + data['U'], \
			data['x_vars'] + data['u_vars'], \
			deg_max)

	# eq (3)
	T_eq3 = -PolyLinTrans.eye(num_x_var, num_x_var, deg_rho)
	Aeq31, Aeq32, beq3, pos_sdsos3 = add_sdsos(T_eq3, -1+data['tol'], \
			data['K'], \
			data['x_vars'], \
			deg_max)

	Aeq = scipy.sparse.bmat([[ Aeq11, Aeq12, None], [Aeq31, None, Aeq32]], format='coo') 
	beq = np.hstack([beq1, beq3])

	pos_sdsos = [(Aeq11.shape[1] + pos, length) for (pos, length) in pos_sdsos1] + \
				[(Aeq11.shape[1] + Aeq12.shape[1] + pos, length) for (pos, length) in pos_sdsos3]

	# eq(2)
	T_eq2 = PolyLinTrans.eye(num_x_var, num_x_var, deg_rho)
	for XK_set_i in XK_set:
		Aeq21, Aeq22, beq2, pos_sdsos2 = add_sdsos(T_eq2, data['tol'], \
				XK_set_i, \
				data['x_vars'], \
				deg_max)
		Aeq21 = scipy.sparse.bmat([[Aeq21, _coo_zeros(Aeq21.shape[0], Aeq.shape[1] - Aeq21.shape[1])]])

		pos_sdsos += [(Aeq.shape[1] + pos, length) for (pos, length) in pos_sdsos2]

		Aeq = scipy.sparse.bmat([[Aeq, None], [Aeq21, Aeq22]])
		beq = np.hstack([beq, beq2])


	# objective
	int_trans = PolyLinTrans.integrate(num_x_var, deg_rho, range(num_x_var), [[-1,1]]*num_x_var)
	_, _, _, obj_idx, obj_vals = int_trans.to_sparse()

	print "completed in " + str(time.clock() - t_start) + "\n"
	print "setting up Mosek SOCP..."
	t_start = time.clock()

	# Set up mosek environment
	env = mosek.Env() 

	task = env.Task(0,0)

	numcon = Aeq.shape[0]
	numvar = Aeq.shape[1]

	task.appendvars(numvar) 
	task.appendcons(numcon)

	# Make all vars unbounded
	task.putvarboundslice(0, numvar, [mosek.boundkey.fr] * numvar, [0.]*numvar, [0.]*numvar )

	# Add objective function
	for (i,v) in zip(obj_idx, obj_vals):
		task.putcj(i, v)
	task.putobjsense(mosek.objsense.maximize) 

	# Add eq constraints
	task.putaijlist( Aeq.row, Aeq.col, Aeq.data )
	task.putconboundslice(0, numcon, [mosek.boundkey.fx] * numcon, beq, beq )

	# Add sdd constraints
	for pos, length in pos_sdsos:
		add_sdd_mosek( task, pos, length )

	print "completed in " + str(time.clock() - t_start) + "\n"

	print "optimizing..."
	t_start = time.clock()

	task.optimize() 

	solsta = task.getsolsta(mosek.soltype.itr)

	if (solsta == mosek.solsta.optimal or solsta == mosek.solsta.near_optimal): 
		print "completed in " + str(time.clock() - t_start) + "\n"
		print "status", solsta
		# extract solution
		opt_rho = np.zeros(Aeq11.shape[1])
		task.getxxslice( mosek.soltype.itr, 0, Aeq11.shape[1], opt_rho )

		return coef_to_poly(opt_rho, data['x_vars'])

	else: 
		print "Optimal solution not found"
		print solsta
		return None
