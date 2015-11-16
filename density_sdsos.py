import numpy as np
import sympy as sp

import sys
from mosek.fusion import *
from mosek.array import *

from polylintrans import *

from sdd import setup_sdd, is_sdd, get_symmetric_matrix

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
	ret = 0
	for k in range(len(coef)):
		midx = index_to_multiindex(k, num_var)
		mon = coef[k]*sp.Symbol('t')**midx[0]
		for i in range(1,num_var):
			mon *= sp.Symbol('x%d' % i)**midx[i]
		ret += mon
	return ret

def degree(polytuple):
	if isinstance(polytuple, list):
		if len(polytuple) == 0:
			return 0
		else:
			return max([degree(p) for p in polytuple])
	return max([sum(term[0]) for term in polytuple])

def Lf(d, vf):
	""" Linear transformation representing Lf operator on a polynomial of degree d """
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