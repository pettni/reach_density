""" 
 	Collection of methods that are useful for dealing with optimization 
 	over (scaled) diagonally dominant sums of squares polynomials.
"""

import sympy as sp
import numpy as np

import sys
import picos
from mosek.fusion import *

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

def setup_sdd_picos(prob, var, sdd_str = ''):
	""" Make sure that the expression matVar is sdd by adding constraints to the model M.
		Additional variables Mij of size n*(n-1)/2 x 3 are required, where  each row represents a symmetric
		2x2 matrix
	    Mij(k,:) is the vector Mii Mjj Mij representing [Mii Mij; Mij Mjj] for (i,j) = _k_to_ij(k)"""

	# Length of vec(M)
	num_vars = var.size[0]

	if num_vars == 1:
		# scalar case
		prob.add_constraint('', var >= 0)
		return None

	# Side of M
	n = int((np.sqrt(1+8*num_vars) - 1)/2)
	print num_vars
	print n

	assert(n == (np.sqrt(1+8*num_vars) - 1)/2)

	# Number of submatrices required
	num_Mij = n*(n-1)/2

	Mij = prob.add_variable('Mij_' + sdd_str, (num_Mij, 3))

	# add pos and cone constraints ensuring that each Mij(k,:) is psd
	prob.add_list_of_constraints( [Mij[k,0] >= 0 for k in range(num_Mij)], 'k', '[' + str(num_Mij) + ']' )
	prob.add_list_of_constraints( [Mij[k,1] >= 0 for k in range(num_Mij)], 'k', '[' + str(num_Mij) + ']' )
	prob.add_list_of_constraints( [Mij[k,0] * Mij[k,1] >= Mij[k,2]**2 for k in range(num_Mij)], 'k', '[' + str(num_Mij) + ']' )

	# set Aij = Mij 
	for i in range(n):
		for j in range(i,n):
			A_idx = _ij_to_k(i,j,num_vars)
			M_idx = _sdd_index(i,j,n)
			prob.add_constraint( var[A_idx] == picos.sum( [ Mij[k,l] for (k,l) in M_idx ] ) )

	# prob.add_list_of_constraints( [ 
	# 		var[ _ij_to_k(i,j,num_vars) ] == 
	# 		picos.sum( [ Mij[k,l] for k,l in _sdd_index(i,j,n) ], 'k,l', '_sdd_index(i,j,n)')
	# 			for i in range(n) for j in range(i,n) 
	# 	], 'i,j', 'i,j : 0 <= i <= j < n' )

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