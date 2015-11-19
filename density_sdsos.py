import numpy as np
import sympy as sp
import scipy.sparse 

import sys
import mosek
from mosek.fusion import *
from mosek.array import *

from polylintrans import *

import picos
import cvxopt as cvx

from sdd import setup_sdd, setup_sdd_picos, setup_sdd_mosek

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
	for k in range(len(coef)):
		midx = index_to_multiindex(k, num_var)
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

def compute_reach(data):
	deg_max = data['maxdeg_rho']
	rho_0 = poly_to_tuple(data['rho_0'], data['variables'])
	vf = poly_to_tuple(data['vector_field'], data['variables'])
	domain = poly_to_tuple(data['domain'], data['variables'])
	r = data['r']
	tol = data['tol']

	assert(deg_max % 2 == 0)

	# Variables are [t x1 x2 ... xn]

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

	print "Using degrees:"
	print "Maximal degree", deg_max
	print "deg(rho) = ", deg_rho
	print "deg(sigma) = ", deg_sigma

	# Compute number of variables  
	n_max_mon = count_monomials_leq(num_var, deg_max)    	 # monomial representation
	n_max_sq = count_monomials_leq(num_var, halfdeg_max) * \
				(count_monomials_leq(num_var, halfdeg_max) + 1)/2  # square matrix representation
	
	n_rho = count_monomials_leq(num_var, deg_rho) 		   # number of terms in rho(t,x)

	n_sigma_sq = count_monomials_leq(num_var, halfdeg_sigma) * \
			     (count_monomials_leq(num_var, halfdeg_sigma) + 1)/2


	########################################################
	### Extract matrices defining linear transformations ###
	########################################################

	# Linear trans b -> coef(b)
	b_b = PolyLinTrans(num_var)
	b_b[(0,)*num_var][(0,)*num_var] = 1.
	nrow, ncol, idxi, idxj, vals = b_b.to_sparse()
	A_b_b = Matrix.sparse(n_max_mon, ncol, idxi, idxj, vals)

	# Linear trans coef(rho) -> coef(Lrho)
	Lrho_rho = Lf(deg_rho, vf)
	nrow, ncol, idxi, idxj, vals = Lrho_rho.to_sparse()
	A_Lfrho_rho = Matrix.sparse(n_max_mon, ncol, idxi, idxj, vals)

	# Linear trans coef(s_i) -> coef(s_i d_i)
	sidi_si = [PolyLinTrans.mul_pol(num_var, deg_sigma, dom) for dom in domain]
	A_sidi_si = []
	for i in range(len(domain)):
		nrow, ncol, idxi, idxj, vals = sidi_si[i].to_sparse_matrix()
		A_sidi_si.append(Matrix.sparse(n_max_mon, ncol, idxi, idxj, vals))

	# Get identity transformation w.r.t vector representing symmetric matrix
	nrow, ncol, idxi, idxj, vals = PolyLinTrans.eye(num_var, deg_max).to_sparse_matrix()
	A_poly_K = Matrix.sparse(n_max_mon, ncol, idxi, idxj, vals)

	# Transformation rho -> rho0
	rho0_rho = PolyLinTrans.elvar(num_var, deg_rho, 0, 0)
	nrow, ncol, idxi, idxj, vals = rho0_rho.to_sparse()
	A_rho0_rho = Matrix.sparse(nrow, ncol, idxi, idxj, vals)

	# Convert initial poly to sparse vector
	b_idxi = [multiindex_to_index(arg[0]) for arg in rho_0]
	b_vali = [arg[1] for arg in rho_0]
	b_idxj = [0]*len(b_idxi)
	b_rho0 = Matrix.sparse(nrow, 1, b_idxi, b_idxj, b_vali)

	error = np.inf
	sln = 0

	with Model('test') as M: 

		# Optimization variables are introduced as follows
		#
		#  rho 				: coefficients for density - monomial vector
		#  b   				: bound - scalar
		#  si_vars_neg[]  	: sdd multipliers - vector representing symmetric matrix
		#  si_vars_pos[]  	: sdd multipliers - vector representing symmetric matrix
		#  Kpos				: dummy variable  - vector representing symmetric matrix
		#  Kneg				: dummy variable  - vector representing symmetric matrix
		#

		########################
		### Define variables ###
		########################

		b = M.variable(1, Domain.greaterThan(0.))	# bound
		rho = M.variable(n_rho, Domain.unbounded())	# density coefficients
		si_vars_pos = []							# positive multipliers
		si_vars_neg = []							# positive multipliers
		for i in range(len(domain)):
			si_vars_pos.append(M.variable(n_sigma_sq, Domain.unbounded()))
			si_vars_neg.append(M.variable(n_sigma_sq, Domain.unbounded()))

		Kpos = M.variable(n_max_sq, Domain.unbounded())  # positive dummy var
		Kneg = M.variable(n_max_sq, Domain.unbounded())  # positive dummy var

		########################
		##### Constraints ######
		########################

		# make multipliers sdd
		for i in range(len(domain)):
			setup_sdd(M, si_vars_pos[i])
			setup_sdd(M, si_vars_neg[i])

		# make K's sdd
		setup_sdd(M, Kpos)
		setup_sdd(M, Kneg)

		# Stacked sparse matrices [A_b_b A_Lfrho_rho A_sidi_si A_poly_K]
		A_combined = Matrix.sparse( [[A_b_b, A_Lfrho_rho] + A_sidi_si + [A_poly_K]])

		# Stacked variables
		vars_combined_pos = Expr.vstack( [b.asExpr(), rho.asExpr()] + [Expr.neg(si) for si in si_vars_pos] + [Expr.neg(Kpos)])
		vars_combined_neg = Expr.vstack([b.asExpr(), Expr.neg(rho)] + [Expr.neg(si) for si in si_vars_neg] + [Expr.neg(Kneg)])

		# Set b + L rho - s_i^+ d_i = K+
		M.constraint( Expr.mul(A_combined, vars_combined_pos) , Domain.equalsTo( 0. ) )

		# Set b - L rho - s_i^+ d_i = K-
		M.constraint( Expr.mul(A_combined, vars_combined_neg) , Domain.equalsTo( 0. ) )

		# Stack variables for initial constraint
		A0_combined = Matrix.sparse([[A_rho0_rho, b_rho0]])
		vars0_combined = Expr.vstack([rho.asExpr(), Expr.constTerm(-1.)])

		# initial constraint
		M.constraint(Expr.mul(A0_combined, vars0_combined), Domain.equalsTo(0.))

		### Objective ###
		M.objective(ObjectiveSense.Minimize, b)

		# Enable logger output
		M.setLogHandler(sys.stdout) 

		# Solve it!
		M.solve()

		error = b.level()[0]

		rho_coef_trimmed = [c if abs(c) > tol else 0 for c in rho.level()] 
		rho_sln = coef_to_poly(rho_coef_trimmed, num_var)

	return (rho_sln, error)

def compute_reach_picos(data):
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

	print "Using degrees:"
	print "Maximal degree", deg_max
	print "deg(rho) = ", deg_rho
	print "deg(sigma) = ", deg_sigma

	# Compute number of variables  
	n_max_mon = count_monomials_leq(num_var, deg_max)    	 # monomial representation
	n_max_sq = count_monomials_leq(num_var, halfdeg_max) * \
				(count_monomials_leq(num_var, halfdeg_max) + 1)/2  # square matrix representation
	
	n_rho = count_monomials_leq(num_var, deg_rho) 		   # number of terms in rho(t,x)

	n_sigma_sq = count_monomials_leq(num_var, halfdeg_sigma) * \
			     (count_monomials_leq(num_var, halfdeg_sigma) + 1)/2


	##########################################################
	### Construct matrices defining linear transformations ###
	##########################################################

	# Linear trans b -> coef(b)
	b_b = PolyLinTrans(num_var)
	b_b[(0,)*num_var][(0,)*num_var] = 1.
	nrow, ncol, idxi, idxj, vals = b_b.to_sparse()
	A_b_b =  cvx.spmatrix( vals, idxi, idxj, (n_max_mon, ncol) )

	# Linear trans coef(rho) -> coef(Lrho)
	Lrho_rho = Lf(deg_rho, vf)
	nrow, ncol, idxi, idxj, vals = Lrho_rho.to_sparse()
	A_Lfrho_rho =  cvx.spmatrix( vals, idxi, idxj, (n_max_mon, ncol) )

	# Linear trans coef(s_i) -> coef(s_i d_i)
	sidi_si = [PolyLinTrans.mul_pol(num_var, deg_sigma, dom) for dom in domain]
	A_sidi_si = []
	for i in range(len(domain)):
		nrow, ncol, idxi, idxj, vals = sidi_si[i].to_sparse_matrix()
		A_sidi_si.append( cvx.spmatrix( vals, idxi, idxj, (n_max_mon, ncol) ) )

	# Get identity transformation w.r.t vector representing symmetric matrix
	nrow, ncol, idxi, idxj, vals = PolyLinTrans.eye(num_var, deg_max).to_sparse_matrix()
	A_poly_K = cvx.spmatrix(vals, idxi, idxj, (n_max_mon, ncol))

	# Transformation rho -> rho0
	rho0_rho = PolyLinTrans.elvar(num_var, deg_rho, 0, 0)
	nrow, ncol, idxi, idxj, vals = rho0_rho.to_sparse()
	A_rho0_rho =   cvx.spmatrix( vals, idxi, idxj, (nrow, ncol) )
	
	beq = cvx.matrix(0., (A_rho0_rho.size[0],1) )
	for arg in rho_0:
		beq[multiindex_to_index(arg[0]),0] = arg[1]

	# Constraints
	#    rho(0,x) = rho0(x) 					(1)
	#    Lf rho + b - sigma_i^+ d_i  sdd		(2)
	#   -Lf rho + b - sigma_i^- d_i  sdd		(3)
	#   sigma_i^+ sdd 							(6)
	#   sigma_i^- sdd 							(7)

	# Define optimization problem
	prob = picos.Problem()

	# Add variables
	rho = prob.add_variable( 'rho', n_rho )
	b = prob.add_variable( 'b', 1 )
	K_pos = prob.add_variable( 'K_pos', n_max_sq)
	K_neg = prob.add_variable( 'K_neg', n_max_sq)
	sig_pos_i = [ prob.add_variable( 'sig_pos[%d]' %i, n_sigma_sq ) for i in range(len(domain)) ]
	sig_neg_i = [ prob.add_variable( 'sig_neg[%d]' %i, n_sigma_sq ) for i in range(len(domain)) ]

	# Add init constraint
	prob.add_constraint( A_rho0_rho*rho == beq )

	# Add eq constraints
	prob.add_constraint(  A_Lfrho_rho * rho + A_b_b * b 
					- picos.sum([A_sidi_si[i] * sig_pos_i[i] for i in range(len(domain))], 'i', '[' + str(len(domain)) + ']') 
					== A_poly_K * K_pos
				)
	prob.add_constraint( -A_Lfrho_rho * rho + A_b_b * b 
					- picos.sum([A_sidi_si[i] * sig_neg_i[i] for i in range(len(domain))], 'i', '[' + str(len(domain)) + ']') 
					== A_poly_K * K_neg
				)

	# Make Ks sdd
	sdd_pos = setup_sdd_picos( prob, K_pos, 'K_pos')
	sdd_neg = setup_sdd_picos( prob, K_neg, 'K_neg')

	# Make sigmas sdd
	for i in range(len(domain)):
		setup_sdd_picos( prob, sig_pos_i[i], 'sig_pos_' + str(i))
		setup_sdd_picos( prob, sig_neg_i[i], 'sig_neg_' + str(i))

	# Minimize b
	prob.set_objective('min', b)

	prob.solve(solver='cvxopt', verbose=True)

	return (coef_to_poly(np.array(rho.value), num_var)[0], np.array(b.value)[0][0])

def _coo_zeros(nrow,ncol):
	'''
	Create a scipy.sparse zero matrix with 'nrow' rows and 'ncol' columns
	'''
	return scipy.sparse.coo_matrix( (nrow,ncol) )

def compute_reach_fast(data):
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

	print "Using degrees:"
	print "Maximal degree", deg_max
	print "deg(rho) = ", deg_rho
	print "deg(sigma) = ", deg_sigma

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

	# vars required for sdd repr. of overall eq, sigma
	n_max_sddvar = 3 * count_monomials_leq(num_var, halfdeg_max) * \
				(count_monomials_leq(num_var, halfdeg_max) - 1)/2
	n_sigma_sddvar = 3 * count_monomials_leq(num_var, halfdeg_sigma) * \
			     (count_monomials_leq(num_var, halfdeg_sigma) - 1)/2

	##########################################################
	### Construct matrices defining linear transformations ###
	##########################################################

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
	A_sd_s_stacked = scipy.sparse.bmat([ A_sidi_si ])

	# Get identity transformation w.r.t vector representing symmetric matrix
	nrow, ncol, idxi, idxj, vals = PolyLinTrans.eye(num_var, deg_max).to_sparse_matrix()
	A_poly_K = scipy.sparse.coo_matrix((vals, (idxi, idxj)), shape = (n_max_mon, ncol))

	# Transformation rho -> rho0
	rho0_rho = PolyLinTrans.elvar(num_var, deg_rho, 0, 0)
	nrow, ncol, idxi, idxj, vals = rho0_rho.to_sparse()
	A_rho0_rho = scipy.sparse.coo_matrix( (vals, (idxi, idxj)), shape = (nrow, ncol) )
	
	beq = np.zeros(A_rho0_rho.shape[0])
	for arg in rho_0:
		beq[multiindex_to_index(arg[0])] = arg[1]

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

	numvar = n_rho + 1 + 2 * len(domain) * n_sigma_sq + 2 * n_max_sq

	# Constraints
	# 
	#   A_rho0_rho * rho = beq  						
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

	Aeq = scipy.sparse.bmat([
		[ scipy.sparse.bmat( [[ A_rho0_rho, _coo_zeros(n_con_init, numvar - n_rho) ]] ) ],
		[ scipy.sparse.bmat( [[  A_Lfrho_rho, A_b_b, -A_sd_s_stacked, _coo_zeros(n_con_K, len(domain) * n_sigma_sq ), -A_poly_K, _coo_zeros(n_con_K, n_max_sq) ]] ) ],
		[ scipy.sparse.bmat( [[ -A_Lfrho_rho, A_b_b, _coo_zeros(n_con_K, len(domain) * n_sigma_sq ), -A_sd_s_stacked, _coo_zeros(n_con_K, n_max_sq), -A_poly_K ]] ) ]
	])
	beq = np.hstack([beq, np.zeros(2 * n_con_K)])

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

	# Add constraints
	Aeq = Aeq.tocoo()
	task.putaijlist( Aeq.row, Aeq.col, Aeq.data )
	task.putconboundslice(0, numcon, [mosek.boundkey.fx] * numcon, beq, beq )

	for j in range(len(domain)):
		start = n_rho + 1 + j * n_sigma_sq
		length = n_sigma_sq
		setup_sdd_mosek( task, start, length )								# make sigma_i^+ sdd
		setup_sdd_mosek( task, start + len(domain) * n_sigma_sq, length) 	# make sigma_i^- sdd

	start = n_rho + 1 + 2 * len(domain) * n_sigma_sq
	length = n_max_sq
	setup_sdd_mosek( task, start, length )			# make K^+ sdd
	setup_sdd_mosek( task, start+length, length) 	# make K^- sdd
	
	numvar_sdd = 2 * len(domain) * n_sigma_sddvar + 2 * n_max_sddvar

	task.optimize() 

	opt_rho = np.zeros(n_rho)
	opt_err = np.zeros(1)
	task.getxxslice( mosek.soltype.itr, 0, n_rho, opt_rho )
	task.getxxslice( mosek.soltype.itr, n_rho, n_rho+1, opt_err )

	return ( coef_to_poly(opt_rho, num_var), opt_err[0])