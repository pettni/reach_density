import sympy as sp
import numpy as np

import sys
from mosek.fusion import *
from mosek.array import *

from sympy.polys.monomials import itermonomials, monomial_count
from sympy.polys.orderings import monomial_key

from help_functions import _compute_Lrho, _extract_linear

def compute_density(data):

	maxdeg = data['maxdeg']
	variables = data['variables']
	rho_0_given = data['rho_0']
	vector_field = data['vector_field']

	xdim = len(variables) - 1

	assert(xdim == len(vector_field))

	n = monomial_count(xdim+1, maxdeg)

	print "There are ", n, " variables"

	# define rho with arbitrary coefficients
	monomials = sorted(itermonomials(variables, maxdeg), key=monomial_key('lex', variables))
	cfs = [sp.Symbol('a_%d' % i) for i in range(n)]
	rho = np.dot(cfs, monomials)

	result = _compute_Lrho(rho, vector_field, variables)

	# Extract linear transformation
	Ae, be = _extract_linear(sp.Poly(result, variables).coeffs(), cfs)

	# Compute initial distribution
	rho_0 = rho.subs('t', 0)

	Ae0, be0 = _extract_linear(sp.Poly(rho_0-rho_0_given, variables).coeffs() ,cfs)

	error = np.inf
	sln = 0

	with Model('test') as M: 
		a = M.variable("a", int(n), Domain.unbounded())
		t = M.variable("t", 1, Domain.greaterThan(0.))

		# minimize maximal Liouville coefficient
		ones = [1. for i in range(len(Ae))]
		M.constraint("t1", Expr.sub( Expr.mul(t, ones) ,Expr.mul(DenseMatrix(Ae),a) ), Domain.greaterThan(0.) )
		M.constraint("t2", Expr.add( Expr.mul(t, ones) ,Expr.mul(DenseMatrix(Ae),a) ), Domain.greaterThan(0.) )

		# add initial constraint
		M.constraint("init", Expr.mul(DenseMatrix(Ae0), a), Domain.equalsTo(be0))

		M.objective("minvar", ObjectiveSense.Minimize, t)

		M.solve()

		error = t.level()[0]*n
		sln = np.dot(a.level(), monomials)

	return (sln, error)

