""" 
 	Collection of methods that are useful for dealing with optimization 
 	over (scaled) diagonally dominant sums of squares polynomials.
"""

import sympy as sp
import numpy as np

import sys
from mosek.fusion import *

from sympy.abc import x,y,z

from sympy.polys.monomials import itermonomials, monomial_count
from sympy.polys.orderings import monomial_key

from help_functions import extract_linear, flatten

def _k_to_ij(k, L):
	""" 
	Given a symmetric matrix Q represented by a vector
		V = [Q_00, ... Q_0n, Q_11, ..., Q_1n] of length L,
		for given k, compute i,j s.t. Q_ij = V(k)
	"""

	if (k >= L):
		raise IndexError("Index out of range")

	# inverse formula for arithmetic series
	n = (np.sqrt(1+8*L) - 1)/2

	# get first index
	i = int(np.ceil( (2*n+1)/2 - np.sqrt( ((2*n+1)/2)**2 -2 * (k+1)  ) ) - 1)

	# second index
	k1 = (2*n+1-i)*i/2 - 1
	j = int(i + k - k1 - 1)

	return i,j

def _ij_to_k(i,j,L):
	# Given a symmetric matrix Q represented by a vector
	# V = [Q_00, ... Q_0n, Q_11, ..., Q_1n] of length L,
	# for given i,j , compute k s.t. Q_ij = V(k)
	n = (np.sqrt(1+8*L) - 1)/2
	i_at1 = min(i,j)+1
	j_at1 = max(j,i)+1
	k_at1 = int((n + n-i_at1)*(i_at1-1)/2 + j_at1)
	return k_at1 - 1

def degree(expr, variables):
	"""Check the degree of variables 'variables' in an expression expr.
		Warning: if the expression is factored, it will be expanded.
				 this might be expensive.
	"""
	if isinstance(expr, list):
		return max([degree(el, variables) for el in expr])
	expr_ex = expr.expand()
	var_idx = [i for i in range(len(expr_ex.as_terms()[1])) if expr_ex.as_terms()[1][i] in variables]
	return max([sum(term[1][1][idx] for idx in var_idx) for term in expr_ex.as_terms()[0] ])

def create_square_poly(d, variables, coef_name):
	num_mons = monomial_count(d, len(variables))
	num_coef = num_mons*(num_mons+1)/2
	monomials = sorted(itermonomials(variables, d), key=monomial_key('lex', variables))
	A = [sp.Symbol(coef_name + '_%d' % i) for i in range(num_coef)]
	poly = sp.expand(np.dot(np.dot(get_symmetric_matrix(A), monomials), monomials))
	return poly, A

def create_square_poly_var(M, d, num_vars):
	num_mons = monomial_count(d, len(variables))
	num_coef = num_mons*(num_mons+1)/2
	return M.variable(num_coef, Domain.unbounded())

def get_symmetric_matrix(V):

	L = len(V)
	n = (np.sqrt(1+8*L) - 1)/2

	if not n.is_integer():
		raise TypeError("vector does not represent symmetric matrix")

	n = int(n)

	Q = [[None for i in range(n)] for j in range(n) ]

	for k in range(L):
		i,j = _k_to_ij(k, L)
		Q[i][j] = Q[j][i] = V[k]

	return Q

def _sdd_index(i,j,n):
	""" An n x n sdd matrix A can be written as A = sum Mij.
		Given Mij's stored as a (n-1)*n/2 x 3 matrix, where each row represents a 2x2 symmetric matrix, return
	    the indices i_s, j_s such that A_ij = sum_s Mij(i_s, j_s) """
	num_vars = int(n*(n-1)/2)
	if i == j:
		return [ [_ij_to_k(min(i,l), max(i,l)-1, num_vars),(0 if i<l else 1)] for l in range(n) if l != i ]
	else:
		return [[_ij_to_k(min(i,j), max(i,j)-1, num_vars),2]]

def setup_sdd(M, matVar):
	""" Make sure that the expression matVar is sdd by adding constraints to the model M.
		Additional variables Mij of size n*(n-1)/2 x 3 are required, where  each row represents a symmetric
		2x2 matrix
	    Mij(k,:) is the vector Mii Mjj Mij representing [Mii Mij; Mij Mjj] for (i,j) = _k_to_ij(k)"""

	num_vars = int(matVar.size())
	if num_vars == 1:
		# scalar case
		M.constraint(matVar, Domain.greaterThan(0.))
		return None

	n = int((np.sqrt(1+8*num_vars) - 1)/2)

	num_Mij = n*(n-1)/2

	Mij = M.variable(Set.make(num_Mij, 3), Domain.unbounded())

	# add pos and cone constraints ensuring that each Mij(k,:) is psd
	for k in range(num_Mij):
		M.constraint(Mij.index(k,0), Domain.greaterThan(0.))
		M.constraint(Mij.index(k,1), Domain.greaterThan(0.))
		# (x,y,z) in RotatedQCone <--> xy/2 >= z**2
		Mij_expr = Expr.vstack(Expr.mul(0.5, Mij.index(k,0)), Mij.pick([[k,j] for j in range(1,3)]))
		M.constraint(Mij_expr, Domain.inRotatedQCone(3))
	
	# set Aij = Mij for i != j
	for i in range(n):
		for j in range(i,n):
			A_idx = _ij_to_k(i,j,num_vars)
			M_idx = _sdd_index(i,j,n)
			M.constraint(Expr.sub(matVar.index(A_idx), Expr.sum(Mij.pick(M_idx) ) ), Domain.equalsTo(0.0) )
	return Mij
			
def is_dd(A):
	""" Returns 'True' if A is dd (diagonally dominant), 'False' otherwise """

	epsilon = 1e-10 # have some margin

	A_arr = np.array(A)
	if A_arr.shape[0] != A_arr.shape[1]:
		return False

	n = A_arr.shape[0]
	for i in range(n):
		if not A[i,i] + epsilon >= sum(np.abs(A[i, [j for j in range(n) if i != j]])):
			return False

	return True

def is_sdd(A):
	""" Returns 'True' if A is sdd (scaled diagonally dominant), 'False' otherwise """

	epsilon = 1e-5

	A_arr = np.array(A)
	n = A_arr.shape[0]

	# Define a LP
	M = Model()

	Y = M.variable(n, Domain.greaterThan(1.))
	for i in range(n):
		K_indices = [i,[j for j in range(n) if i != j]]
		Y_indices = [j for j in range(n) if i != j]
		M.constraint( Expr.sub( Expr.mul(A_arr[i,i], Y.index(i) ), 
					            Expr.dot(np.abs(A_arr[K_indices]).tolist(), 
					            	Y.pick(Y_indices) )
					          ),
					          Domain.greaterThan(-epsilon)
					) 

	M.objective(ObjectiveSense.Minimize, Expr.sum(Y))

	M.solve()

	return False if M.getDualSolutionStatus() == SolutionStatus.Certificate else True

def is_sdsos(poly, vrs):
	d = int(np.ceil(degree(poly, vrs)/2))
	
	polyA, A = create_square_poly(d, vrs, 'a')

	cfs = sp.Poly(poly - Apoly, vrs).coeffs()
	Ae, be = _extract_linear(cfs, A)

	M = Model('check_sdsos')
	Avar = create_square_poly_var(M, d, len(vrs))
	M.constraint(Expr.mul(DenseMatrix(Ae), Avar), Domain.equalsTo(be))

	M.solve()

def minimize_poly(p, variables, domain, sigma_d):
	""" Lower bounds the minimum value of p in domain by solving

		max  t
		s.t.  p - t - s_i d_i sdsos  for some s_i sdsos
	"""
	t = sp.symbols('t')

	sigma = [None for i in range(len(domain))]
	sigma_coef = [None for i in range(len(domain))]

	for i in range(len(domain)):
		s, s_c = create_square_poly(sigma_d, variables, 'sigma'+str(i))
		sigma[i] = s
		sigma_coef[i] = s_c

	sd_poly = p - t - np.dot(sigma, domain)

	d = int(np.ceil(degree(sd_poly, variables)/2))

	Apoly, A = create_square_poly(d, variables, 'a')

	cfs = sp.Poly(sd_poly - Apoly, variables).coeffs()
	Ae, be = _extract_linear(cfs, [t] + A + _flatten(sigma_coef))
 	with Model('sdsos') as M:
		tvar = M.variable(1, Domain.unbounded())
		Avar = M.variable(len(A), Domain.unbounded())

		# Setup multipliers
		sigma_vars = [M.variable(len(sigma_coef[i]), Domain.unbounded()) for i in range(len(domain))]
		for i in range(len(domain)):
			setup_sdd(M, sigma_vars[i]) # make sl sdd

		if len(domain) > 0:
			sigma_vars_flat = Expr.vstack([var.asExpr() for var in sigma_vars])
			tAs_vars = Expr.vstack(Expr.vstack(tvar, Avar), sigma_vars_flat) 
		else:
			tAs_vars = Expr.vstack(tvar, Avar) 

		# Set sd_poly = Apoly
		M.constraint(Expr.mul(DenseMatrix(Ae), tAs_vars), Domain.equalsTo(be))

		# Make Apoly positive
		Mij = setup_sdd(M, Avar)

		M.objective(ObjectiveSense.Maximize, tvar)

		M.solve()
		monomials = sorted(itermonomials(variables, d), key=monomial_key('lex', variables))
		return tvar.level()[0], sp.expand(np.dot(np.dot(get_symmetric_matrix(Avar.level()), monomials), monomials)), [sig.level() for sig in sigma_vars]