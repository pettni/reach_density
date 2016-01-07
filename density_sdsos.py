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

	# we don't know how to handle sets defined by multiple inequalities
	assert(len(data['K_set']) == 1)

	t_domain = [data['t_var'][0]*(t_f-data['t_var'][0])];

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

	# Extract domains in different variables
	########################################

	## eq (1)
	g_T_txd = poly_to_tuple(t_domain, data['t_var'] + data['x_vars'] + data['d_vars'])
	g_K_txd = poly_to_tuple(data['K_set'], data['t_var'] + data['x_vars'] + data['d_vars'])
	g_D_txd = poly_to_tuple(data['d_domain'], data['t_var'] + data['x_vars'] + data['d_vars'])

	# eq (2), (4)
	g_K_x = poly_to_tuple(data['K_set'], data['x_vars'])

	# eq (3)
	g_T_tx = poly_to_tuple(t_domain, data['t_var'] + data['x_vars'])
	g_XK_tx = poly_to_tuple([a * b for a in data['K_set'] for b in data['x_domain']], data['t_var'] + data['x_vars'])


	# Compute degrees required
	########################################
	def even(a) : return 2*np.floor(a/2)

	# eq (1)
	deg_eq1 = deg_max
	deg_rho = deg_eq1 + 1 - degree(vf)
	deg_s_T_txd = [even(deg_eq1 - degree(g)) for g in g_T_txd]
	deg_s_K_txd = [even(deg_eq1 - degree(g)) for g in g_K_txd]
	deg_s_D_txd = [even(deg_eq1 - degree(g)) for g in g_D_txd]

	# eq (2)
	deg_eq2 = deg_max
	deg_s_K_1x = [even(deg_eq2 - degree(g)) for g in g_K_x]

	# eq (3)
	deg_eq3 = deg_max
	deg_s_T_tx = [even(deg_eq3 - degree(g)) for g in g_T_tx]
	deg_s_XK_tx = [even(deg_eq3 - degree(g)) for g in g_XK_tx]

	# eq (4)
	deg_eq4 = deg_max
	deg_s_K_2x = [even(deg_eq4 - degree(g)) for g in g_K_x]

	# half degrees for symmetric matrix variables
	halfdeg_eq1 = deg_eq1/2
	halfdeg_eq2 = deg_eq2/2
	halfdeg_eq3 = deg_eq3/2
	halfdeg_eq4 = deg_eq4/2

	halfdeg_s_T_txd = [deg/2 for deg in deg_s_T_txd]
	halfdeg_s_K_txd = [deg/2 for deg in deg_s_K_txd]
	halfdeg_s_D_txd = [deg/2 for deg in deg_s_D_txd]
	halfdeg_s_K_1x = [deg/2 for deg in deg_s_K_1x]
	halfdeg_s_T_tx = [deg/2 for deg in deg_s_T_tx]
	halfdeg_s_XK_tx = [deg/2 for deg in deg_s_XK_tx]
	halfdeg_s_K_2x = [deg/2 for deg in deg_s_K_2x]

	# Compute number of variables required
	########################################

	def square_form_count(numvar, halfdeg) : return count_monomials_leq(numvar, halfdeg) * \
				(count_monomials_leq(numvar, halfdeg) + 1)/2

	# rho(t,x)
	n_rho_mon = count_monomials_leq(num_t_var + num_x_var, deg_rho)

	# eq (1)
	n_eq1_mon = count_monomials_leq(num_t_var + num_x_var + num_d_var, deg_eq1)
	n_eq1_sq = square_form_count(num_t_var + num_x_var + num_d_var, halfdeg_eq1)

	# eq (2) 
	n_eq2_mon = count_monomials_leq( num_x_var, deg_eq2)
	n_eq2_sq = square_form_count( num_x_var, halfdeg_eq2)

	# eq (3)
	n_eq3_mon = count_monomials_leq( num_t_var + num_x_var, deg_eq3)
	n_eq3_sq = square_form_count( num_t_var + num_x_var, halfdeg_eq3)

	# eq (4)
	n_eq4_mon = count_monomials_leq(num_x_var, deg_eq4)
	n_eq4_sq = square_form_count(num_x_var, halfdeg_eq4)

	# multipliers
	n_s_T_txd_sq = [square_form_count(num_t_var + num_x_var + num_d_var, deg) for deg in halfdeg_s_T_txd]
	n_s_K_txd_sq = [square_form_count(num_t_var + num_x_var + num_d_var, deg) for deg in halfdeg_s_K_txd]
	n_s_D_txd_sq = [square_form_count(num_t_var + num_x_var + num_d_var, deg) for deg in halfdeg_s_D_txd]
	n_s_K_1x_sq =  [square_form_count(num_x_var, deg) for deg in halfdeg_s_K_1x]
	n_s_T_tx_sq =  [square_form_count(num_t_var + num_x_var, deg) for deg in halfdeg_s_T_tx]
	n_s_XK_tx_sq = [square_form_count(num_t_var + num_x_var, deg) for deg in halfdeg_s_XK_tx]
	n_s_K_2x_sq =  [square_form_count(num_x_var, deg) for deg in halfdeg_s_K_2x]

	
	# Construct matrices defining linear transformations
	#######################################################

	print "setting up matrices..."
	t_start = time.clock()

	# eq (1)

	## Trans sq_coef(rho) [in t,x] -> coef(Lrho) [in t,x,d]
	Lrho_rho = Lf(deg_rho, vf) * PolyLinTrans.eye(num_t_var + num_x_var, num_t_var + num_x_var + num_d_var, deg_rho)
	nrow, ncol, idxi, idxj, vals = Lrho_rho.to_sparse()
	A_Lf1 =  scipy.sparse.coo_matrix( (vals, (idxi, idxj)), shape = (n_eq1_mon, ncol) )

	## Trans sq_coef(s_T_txd) -> coef(s_T_txd g_T_txd)
	g_T_txd_mul = [PolyLinTrans.mul_pol(num_t_var + num_x_var + num_d_var, deg, g) \
							for (deg, g) in zip(deg_s_T_txd, g_T_txd) ]
	A_g_T_txd_mul = []
	for i in range(len(g_T_txd_mul)):
		nrow, ncol, idxi, idxj, vals = g_T_txd_mul[i].to_sparse_matrix()
		A_g_T_txd_mul.append( scipy.sparse.coo_matrix( (vals, (idxi, idxj)), shape = (n_eq1_mon, ncol) ) )
	A_gT1 = scipy.sparse.bmat([ A_g_T_txd_mul ])

	## Trans sq_coef(s_K_txd) -> coef(s_K_txd g_k_txd)
	g_K_txd_mul = [PolyLinTrans.mul_pol(num_t_var + num_x_var + num_d_var, deg, g) \
							for (deg, g) in zip(deg_s_K_txd, g_K_txd) ]
	A_g_K_txd_mul = []
	for i in range(len(g_K_txd_mul)):
		nrow, ncol, idxi, idxj, vals = g_K_txd_mul[i].to_sparse_matrix()
		A_g_K_txd_mul.append( scipy.sparse.coo_matrix( (vals, (idxi, idxj)), shape = (n_eq1_mon, ncol) ) )
	A_gK1 = scipy.sparse.bmat([ A_g_K_txd_mul ])

	## Trans sq_coef(s_D_txd) -> coef(s_D_txd g_D_txd)
	g_D_txd_mul = [PolyLinTrans.mul_pol(num_t_var + num_x_var + num_d_var, deg, g) \
							for (deg, g) in zip(deg_s_D_txd, g_D_txd) ]
	A_g_D_txd_mul = []
	for i in range(len(g_D_txd_mul)):
		nrow, ncol, idxi, idxj, vals = g_D_txd_mul[i].to_sparse_matrix()
		A_g_D_txd_mul.append( scipy.sparse.coo_matrix( (vals, (idxi, idxj)), shape = (n_eq1_mon, ncol) ) )
	try:
		A_gD1 = scipy.sparse.bmat([ A_g_D_txd_mul ])
	except Exception, e:
		A_gD1 = None

	## Trans sq_coef(K1) -> mon_coef(K1)
	nrow, ncol, idxi, idxj, vals = PolyLinTrans.eye(num_t_var + num_x_var + num_d_var, num_t_var + num_x_var + num_d_var, deg_eq1).to_sparse_matrix()
	A_K1 = scipy.sparse.coo_matrix((vals, (idxi, idxj)), shape = (n_eq1_mon, ncol))

	# eq (2)

	## Trans rho -> rho0
	rho0_rho = PolyLinTrans.elvar(num_t_var+num_x_var, deg_rho, 0, 0)
	nrow, ncol, idxi, idxj, vals = rho0_rho.to_sparse()
	A_rho02 = scipy.sparse.coo_matrix( (vals, (idxi, idxj)), shape = (n_eq2_mon, ncol) )

	## Trans rho -> rhoT
	rhoT_rho = PolyLinTrans.elvar(num_t_var+num_x_var, deg_rho, 0, t_f)
	nrow, ncol, idxi, idxj, vals = rhoT_rho.to_sparse()
	A_rhoT2 = scipy.sparse.coo_matrix( (vals, (idxi, idxj)), shape = (n_eq2_mon, ncol) )

	## Trans sq_coef(s_K_1x) -> coef(s_K_1x g_K_x)
	g_K_1x_mul = [PolyLinTrans.mul_pol(num_x_var, deg, g) \
							for (deg, g) in zip(deg_s_K_1x, g_K_x) ]
	A_g_K_1x_mul = []
	for i in range(len(g_K_1x_mul)):
		nrow, ncol, idxi, idxj, vals = g_K_1x_mul[i].to_sparse_matrix()
		A_g_K_1x_mul.append( scipy.sparse.coo_matrix( (vals, (idxi, idxj)), shape = (n_eq2_mon, ncol) ) )
	A_gK2 = scipy.sparse.bmat([ A_g_K_1x_mul ])

	## Trans sq_coef(K2) -> mon_coef(K2)
	nrow, ncol, idxi, idxj, vals = PolyLinTrans.eye(num_x_var, num_x_var, deg_eq2).to_sparse_matrix()
	A_K2 = scipy.sparse.coo_matrix((vals, (idxi, idxj)), shape = (n_eq2_mon, ncol))


	# eq (3)

	## id trans
	A_id3 = scipy.sparse.identity(n_eq3_mon)

	## Trans sq_coef(s_T_tx) -> coef(s_T_tx g_T_tx)
	g_T_tx_mul = [PolyLinTrans.mul_pol(num_t_var + num_x_var, deg, g) \
							for (deg, g) in zip(deg_s_T_tx, g_T_tx) ]
	A_g_T_tx_mul = []
	for i in range(len(g_T_tx_mul)):
		nrow, ncol, idxi, idxj, vals = g_T_tx_mul[i].to_sparse_matrix()
		A_g_T_tx_mul.append( scipy.sparse.coo_matrix( (vals, (idxi, idxj)), shape = (n_eq3_mon, ncol) ) )
	A_gT3 = scipy.sparse.bmat([ A_g_T_tx_mul ])

	## Trans sq_coef(s_XK_tx) -> coef(s_XK_tx g_KX_tx)
	g_XK_tx_mul = [PolyLinTrans.mul_pol(num_t_var + num_x_var, deg, g) \
							for (deg, g) in zip(deg_s_XK_tx, g_XK_tx) ]
	A_g_XK_tx_mul = []
	for i in range(len(g_XK_tx_mul)):
		nrow, ncol, idxi, idxj, vals = g_XK_tx_mul[i].to_sparse_matrix()
		A_g_XK_tx_mul.append( scipy.sparse.coo_matrix( (vals, (idxi, idxj)), shape = (n_eq3_mon, ncol) ) )
	A_gXK3 = scipy.sparse.bmat([ A_g_XK_tx_mul ])

	# print len(A_g_XK_tx_mul)
	# assert(False)

	## Trans sq_coef(K3) -> mon_coef(K3)
	nrow, ncol, idxi, idxj, vals = PolyLinTrans.eye(num_t_var + num_x_var, num_t_var + num_x_var, deg_eq3).to_sparse_matrix()
	A_K3 = scipy.sparse.coo_matrix((vals, (idxi, idxj)), shape = (n_eq3_mon, ncol))

	# eq (4)

	## Trans rho -> rho0
	rho0_rho = PolyLinTrans.elvar(num_t_var+num_x_var, deg_rho, 0, 0)
	nrow, ncol, idxi, idxj, vals = rho0_rho.to_sparse()
	A_rho04 = scipy.sparse.coo_matrix( (vals, (idxi, idxj)), shape = (n_eq4_mon, ncol) )

	## Trans sq_coef(s_K_2x) -> coef(s_K_2x g_K_2x)
	g_K_2x_mul = [PolyLinTrans.mul_pol(num_x_var, deg, g) \
							for (deg, g) in zip(deg_s_K_2x, g_K_x) ]
	A_g_K_2x_mul = []
	for i in range(len(g_K_2x_mul)):
		nrow, ncol, idxi, idxj, vals = g_K_2x_mul[i].to_sparse_matrix()
		A_g_K_2x_mul.append( scipy.sparse.coo_matrix( (vals, (idxi, idxj)), shape = (n_eq4_mon, ncol) ) )
	A_gK4 = scipy.sparse.bmat([ A_g_K_2x_mul ])

	## Trans sq_coef(K4) -> mon_coef(K4)
	nrow, ncol, idxi, idxj, vals = PolyLinTrans.eye(num_x_var, num_x_var, deg_eq4).to_sparse_matrix()
	A_K4 = scipy.sparse.coo_matrix((vals, (idxi, idxj)), shape = (n_eq4_mon, ncol))


	# objective function rho(t,x) -> int rho(0,x) dx

	int_trans = PolyLinTrans.integrate(num_x_var, deg_rho, range(num_x_var), [[-1,1]]*num_x_var) * PolyLinTrans.elvar(num_t_var + num_x_var, deg_rho, 0, 0)
	_, _, _, obj_idx, obj_vals = int_trans.to_sparse()

	print "completed in " + str(time.clock() - t_start) + "\n"

	# Variable vector
	# 
	#  [ rho s_T_txd s_K_txd s_D_txd K1 \   
	#    s_K_1x K2 \
	#    s_T_tx s_XK_tx K3 \
	# 	 s_K_2x K4 ]
	#  with dimensions

	numvars = [n_rho_mon, sum(n_s_T_txd_sq), sum(n_s_K_txd_sq), sum(n_s_D_txd_sq), n_eq1_sq, \
			 sum(n_s_K_1x_sq), n_eq2_sq, \
			 sum(n_s_T_tx_sq), sum(n_s_XK_tx_sq), n_eq3_sq, \
			 sum(n_s_K_2x_sq), n_eq4_sq ]
	numvar = sum(numvars)

	print "setting up Mosek SOCP..."
	t_start = time.clock()

	# Constraints

	numcon = n_eq1_mon + n_eq2_mon + n_eq3_mon + n_eq4_mon

	if A_gD1 == None:
		Aeq = scipy.sparse.bmat( 
		[[ A_Lf1, 					-A_gT1, -A_gK1, -A_K1, 	None, 	None, 	None, 	None, 	None, 	None, 	None],
		 [ (A_rho02 - alpha*A_rhoT2), None, 	None, 	None, 	-A_gK2, -A_K2, 	None, 	None, 	None, 	None, 	None],
		 [ -A_id3, 					None, 	None,   None,   None,	None,	-A_gT3, A_gXK3,	-A_K3, 	None, 	None],
		 [ -A_rho04, 				None, 	None, 	None, 	None, 	None,	None,	None,	None,	-A_gK4, -A_K4]],
		 format = 'coo')
	else:
		Aeq = scipy.sparse.bmat( 
		[[ A_Lf1, 					-A_gT1, -A_gK1, -A_gD1, -A_K1, 	None, 	None, 	None, 	None, 	None, 	None, 	None],
		 [ (A_rho02 - alpha*A_rhoT2), None, 	None, 	None, 	None, 	-A_gK2, -A_K2, 	None, 	None, 	None, 	None, 	None],
		 [ -A_id3, 					None, 	None,   None, 	None,   None,	None,	-A_gT3, A_gXK3,	-A_K3, 	None, 	None],
		 [ -A_rho04, 				None, 	None, 	None, 	None, 	None, 	None,	None,	None,	None,	-A_gK4, -A_K4]],
		 format = 'coo')

	assert(Aeq.shape == (numcon, numvar))

	beq = np.zeros(numcon)
	beq[0] = tol
	beq[n_eq1_mon] = tol
	beq[n_eq1_mon + n_eq2_mon] = tol
	beq[n_eq1_mon + n_eq2_mon + n_eq3_mon] = -1 + tol

	# Set up mosek environment
	env = mosek.Env() 
	task = env.Task(0,0)

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
	add_sdd_mosek( task, sum(numvars[:4]), numvars[4] )		    # make K^1 sdd
	add_sdd_mosek( task, sum(numvars[:6]), numvars[6] )		    # make K^2 sdd
	add_sdd_mosek( task, sum(numvars[:9]), numvars[9] )		    # make K^3 sdd
	add_sdd_mosek( task, sum(numvars[:11]), numvars[11] )	    # make K^4 sdd

	for j in range(len(g_T_txd)):
		add_sdd_mosek( task, sum(numvars[:1]) + sum(n_s_T_txd_sq[:j]), n_s_T_txd_sq[j] )	# make s_T_txd sdd

	for j in range(len(g_K_txd)):
		add_sdd_mosek( task, sum(numvars[:2]) + sum(n_s_K_txd_sq[:j]), n_s_K_txd_sq[j] )	# make s_K_txd sdd

	for j in range(len(g_D_txd)):
		add_sdd_mosek( task, sum(numvars[:3]) + sum(n_s_D_txd_sq[:j]), n_s_D_txd_sq[j] )	# make s_D_txd sdd

	for j in range(len(g_K_x)):
		add_sdd_mosek( task, sum(numvars[:5]) + sum(n_s_K_1x_sq[:j]), n_s_K_1x_sq[j] )	# make s_K_1x sdd

	for j in range(len(g_T_tx)):
		add_sdd_mosek( task, sum(numvars[:7]) + sum(n_s_T_tx_sq[:j]), n_s_T_tx_sq[j] )	# make s_T_tx sdd

	for j in range(len(g_XK_tx)):
		add_sdd_mosek( task, sum(numvars[:8]) + sum(n_s_XK_tx_sq[:j]), n_s_XK_tx_sq[j] )	# make s_XK_tx sdd

	for j in range(len(g_K_x)):
		add_sdd_mosek( task, sum(numvars[:10]) + sum(n_s_K_2x_sq[:j]), n_s_K_2x_sq[j] )	# make s_T_txd sdd

	print "completed in " + str(time.clock() - t_start) + "\n"

	print "optimizing..."
	t_start = time.clock()

	task.optimize() 

	print "completed in " + str(time.clock() - t_start) + "\n"

	# extract solution
	opt_rho = np.zeros(n_rho_mon)
	task.getxxslice( mosek.soltype.itr, 0, n_rho_mon, opt_rho )

	return coef_to_poly(opt_rho, data['t_var'] + data['x_vars'])
