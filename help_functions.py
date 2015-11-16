import sympy as sp
import numpy as np

def extract_linear(lincomb, x):
	# Given a linear combination lincomb of the variables in x, extract (A,b) such that
	# A x = b  < -- > lincomb = 0
	A = [ [float(sp.sympify(expr).coeff(var)) for var in x] for expr in lincomb ]
	dp = np.dot(A, x)
	b = [float(-lincomb[i] + dp[i] ) for i in range(len(lincomb))]
	return A, b

def gradient(f, variables):
	return [sp.diff(f, x) for x in variables]

def compute_Lf(rho, vector_field, variables):
	# Compute L rho symbolically as   (grad rho) . [1 vf]
	if variables[0] != sp.Symbol('t'):
		raise ValueError('t must be first in variables vector')
	
	return np.dot(gradient(rho, variables), [1] + vector_field)

def compute_Lg(rho, vector_field_ctrl, variables):
	# Compute Lg rho symbolically as   d_i rho vf_ij

	return np.dot(gradient(rho, variables), vector_field_ctrl)

def flatten(doublelist):
	return [s for sublist in doublelist for s in sublist]

