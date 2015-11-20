"""
	Class PolyLinTrans for handling linear transformations between coefficient vectors,
	that can be used to formulate optimization problems over polynomials without resorting
	to (slow) symbolic computations.
"""

from math import ceil, sqrt

from itertools import chain
from scipy.misc import comb
import scipy.sparse

from sdd import _k_to_ij

def eval_Lintrans(num_var, trans, vec):
	""" Given a vector of numerical values representing monomial coefficients, output the
	    transformed vector """
	res = [0.]* count_monomials_leq(num_var, trans.d1)
	for i in range(len(vec)):
		midx = index_to_grlex(i, num_var)
		for midx2, v in trans[midx].coeffs.iteritems():
			j = grlex_to_index(midx2)
			res[j] += v * vec[i]
	return res

def grlex_comp(midx1, midx2):
	return cmp(sum(midx1), sum(midx2)) or cmp(midx1, midx2)

def grlex_iter(midx):
	'''
	Create an iterator that produces ordered grlex exponents, starting
	with the multiindex 'midx'.

	Example: The iterator grlex_iter( (0,2) ) produces the sequence
		(0,2) (1,2) (2,0) (0,3) (1,2) (2,1) (3,0) ...
	'''

	midx = list(midx)

	assert(min(midx) >= 0)

	if max(midx) == 0:
		right_ptr = 0
	else:
		# find right-most non-zero element
		for i in range(len(midx)-1, -1, -1):
			if midx[i] != 0:
				right_ptr = i
				break

	while True:
		yield tuple(midx)
		if right_ptr == 0:
			midx = [0] * (len(midx) - 1) + [sum(midx) + 1]
			right_ptr = len(midx) - 1
		else:
			at_right_ptr = midx[right_ptr]
			midx[right_ptr] = 0
			midx[right_ptr-1] += 1
			midx[-1] = at_right_ptr - 1
			right_ptr = len(midx) - 1 if at_right_ptr != 1 else right_ptr - 1

def vec_to_grlex(L, dim):
	''' 
	Given vec(A) for a symmetric matrix A, such that len(vec(A)) = L,
	compute the mapping  vec(A) |--> x^T A x 
	to the grlex dim-dimensionsl monomial basis on multiindex - coefficient form.

	Example: [ a b c ] represents the matrix [ a b ; b c], and the corresponding 1d
	polynomial is a + 2b x + c x^2. Thus the mapping can be represented as
	( 1., 2., 1. ), [ (0.), (1.), (2.) ]
	'''
	n = (sqrt(1 + 8 * L) - 1)/2  # side of matrix

	if not ceil(n) == n:
		raise TypeError('wrong size for vec(A) for A symmetric')

	j = 0
	i = 0
	i_moniter = grlex_iter( (0,) * dim )
	j_moniter = grlex_iter( (0,) * dim )
	i_mon = next(i_moniter)
	j_mon = next(j_moniter)

	exp = []
	coef = [] 

	for k in range(L):
		# loop over all elements
		exp += [tuple( [ii+jj for ii,jj in zip(i_mon, j_mon) ] )]
		coef += [1. if i == j else 2.]

		if j == n-1:
			i += 1
			j = i
			i_mon = next(i_moniter)
			j_moniter = grlex_iter( i_mon )
			j_mon = next(j_moniter)
	
		else:
			j_mon = next(j_moniter)
			j += 1
	return coef, exp

def vec_to_sparse_matrix(L, dim):
	'''
	Sparse matrix representation of the mapping
		vec(A) |--> c,
	such that x_1^T A x_1 = c^T x_2,
	using grlex ordering both for both x_1 and x_2
	'''

	# Obtain coef - exp representation of mapping
	coef, exp = vec_to_grlex(L, dim)

	# sort coef and row indices simultaneously w.r.t grlex ordering
	sorted_ks = zip(coef, range(L))
	sorted_ks.sort(cmp = lambda a1, a2: grlex_comp(exp[a1[1]],exp[a2[1]]) )

	# go through to add cols
	mon_iter = grlex_iter( (0,) * dim )

	cols = [  ]
	current_mon = next(mon_iter)
	current_idx = 0
	for vi,ki in sorted_ks:
		while current_mon != exp[ki]:
			current_mon = next(mon_iter)
			current_idx += 1
		cols += [current_idx]

	# unwrap coef, row
	coef, row = zip(*sorted_ks)

	return scipy.sparse.coo_matrix( (coef, (cols, row) ) )

def grlex_to_index(multiindex):
	"""Returns the grlex ordering number for a given multi-index. Can be expensive for large degrees"""
	total_degree = sum(multiindex)
	n = len(multiindex)

	index = count_monomials_leq(n, total_degree - 1)
	remaining_degree = total_degree
	for i in range(n):
		for k in range(multiindex[i]):
			index += _count_monomials_eq(n-1-i, remaining_degree - k)
		remaining_degree -= multiindex[i]
	return index

def index_to_grlex(index, n):
	"""Returns the multi-index of length n for a given grlex ordering number"""
	grlex = [0 for i in range(n)]

	# find sum of grlex
	total_degree = 0
	while count_monomials_leq(n, total_degree) <= index:
		total_degree += 1

	# index among all multiindices with sum total_degree
	rel_idx_search = index - count_monomials_leq(n, total_degree - 1)

	rem_deg = total_degree # keep track of sum of remaining indices
	rel_idx = 0 # keep track of number of multiindices to the left 
	for i in range(n-1):
		for k in range(total_degree+1):
			new_rel_idx = rel_idx + _count_monomials_eq(n-(i+1), rem_deg-k)
			if new_rel_idx > rel_idx_search:
				grlex[i] = k
				rem_deg -= k
				break
			rel_idx = new_rel_idx

	grlex[-1] = rem_deg

	return tuple(grlex)

def count_monomials_leq(n, d):
	'''Number of monomials in n variables of degree less than or equal to d'''
	return int(comb(n+d, d))

def _count_monomials_eq(n, d):
	'''Number of monomials in n variables of degree equal to d'''
	return int(comb(n+d-1, d))

def _iter_idx(pre, n, d):
	""" Iterate everything to the right of "pre" """
	if len(pre) == n-1:
		for di in range(d + 1 - sum(pre)):
			yield pre + (di,)
	else:
		for di in range(d + 1 - sum(pre)):
			for idx in _iter_idx(pre + (di,), n, d):
				yield idx

class PolyLinTransRow(object):
	"""docstring for PolyLinTransRow"""

	def __init__(self, n):
		self.n = n
		self.coeffs = {}

	def __getitem__(self, midx):
		if len(midx) != self.n:
			raise TypeError('Multiindex does not match polynomial dimension')
		try:
			return self.coeffs[midx]
		except KeyError, e:
			return 0.

	def __setitem__(self, midx, val):
		if len(midx) != self.n:
			raise TypeError('Multiindex does not match polynomial dimension')
		self.coeffs[midx] = float(val)

class PolyLinTrans(object):
	"""Class representing a linear transformation of polynomial coefficients"""

	def __init__(self, n):
		self.n = n 		# number of variables
		self.d0 = 0   	# initial degree
		self.d1 = 0		# final degree
		self.cols = {}

	def rows(self):
		return count_monomials_leq(self.n, self.d1)

	def cols(self):
		return count_monomials_leq(self.n, self.d0)

	def eye(n, d):
		'''
		Identity transformation
		'''
		p = PolyLinTrans(n)
		p.d0 = d
		for idx in _iter_idx((), n, d):
			p[idx][idx] = 1.
		p.d1 = d
		return p
	eye = staticmethod(eye)

	def diff(n, d, xi):
		'''
		Differentiation transformation w.r.t variable xi
		'''
		p = PolyLinTrans(n)
		p.d0 = d
		for idx in _iter_idx((), n, d):
			k = idx[xi]
			new_idx = tuple([(idx[i] if i != xi else idx[i]-1) for i in range(n)])
			if min(new_idx) >= 0:
				p[idx][new_idx] = float(k)
		p.d1 = d-1
		return p
	diff = staticmethod(diff)

	def int(n, d, xi):
		'''
		Integration transformation w.r.t variable xi
		'''
		p = PolyLinTrans(n)
		p.d0 = d
		p.d1 = d+1
		for idx in _iter_idx((), n, d):
			k = idx[xi]
			new_idx = tuple([(idx[i] if i != xi else idx[i]+1) for i in range(n)])
			p[idx][new_idx] = 1./(float(k)+1)
		return p
	int = staticmethod(int)

	def elvar(n, d, xi, val):
		'''
		Transformation resulting from setting xi = val (new polynomial has less variables)
		'''
		p = PolyLinTrans(n)
		for idx in _iter_idx((), n, d):
			new_idx = list(idx)
			new_idx[xi] = 0
			p[idx][tuple(new_idx)] += val**idx[xi]
		p.updated()
		return p
	elvar = staticmethod(elvar)

	def mul_pol(n, d, poly):
		'''
		Transformation representing multiplication of a degree d polynomial with a
		polynomial poly represented as ( ( midx1, cf1 ), (midx2, cf2), ... )
		'''
		p = PolyLinTrans(n)
		maxdeg = 0
		for midx, cf in pol:
			for idx in _iter_idx((), n, d):
				new_idx = tuple([idx[i] + midx[i] for i in range(n)])
				p[idx][new_idx] = float(cf)
			maxdeg = max([maxdeg, sum(midx)])
		p.d0 = d
		p.d1 = d + maxdeg
		return p
	mul_pol = staticmethod(mul_pol)

	def integrate(n, d, dims, box):
		'''
		Transformation representing integration over variables in 'dims'
		over a hyperbox 'box'

		Ex: The mapping p(x,y,z) |--> q(x,z) = \int_0^1 p(x,y,z) dy 
			is obtained by mon(q) = A_int * mon(p) for 
			>> A_int = integrate(3,2,[1],[[0,1]])
		'''
		p = PolyLinTrans.eye(n,d)  # start with right-hand identity
		for (i, xi) in enumerate(dims):
			int_trans = PolyLinTrans.int(n,d,xi)
			upper_trans = PolyLinTrans.elvar(n,d+1,xi,box[i][1])
			lower_trans = PolyLinTrans.elvar(n,d+1,xi,box[i][0])
			p = (upper_trans - lower_trans) * int_trans * p
		p.updated() # degrees become misleading here..
		return p
	integrate = staticmethod(integrate)

	def __getitem__(self, midx):
		if len(midx) != self.n:
			raise TypeError('Multiindex does not match polynomial dimension')
		try:
			return self.cols[midx]
		except KeyError, e:
			self.cols[midx] = PolyLinTransRow(self.n)
			return self.cols[midx]

	def __str__(self):
		ret = 'Transformation in ' + str(self.n) + ' variables from degree ' + \
			 str(self.d0) + ' to degree ' + str(self.d1) + ': \n'
		for key1, col in self.cols.iteritems():
			for key2, val in col.coeffs.iteritems():
				ret += str(key1) + ' --> ' + str(key2) + ' : ' + str(val) + '\n'
		return ret

	def __add__(self, other):
		""" Sum of two linear transformations """
		if not self.n == other.n:
			raise TypeError('Dimension mismatch')
		ret = PolyLinTrans(self.n)
		for midx1, col in chain(self.cols.iteritems(), other.cols.iteritems()):
			for midx2, val in col.coeffs.iteritems():
				try:
					ret[midx1][midx2] += val
				except KeyError, e:
					# didn't exist yet
					ret[midx1][midx2] = val
		ret.d0 = max(self.d0, other.d0)
		ret.d1 = max(self.d1, other.d1)
		return ret

	def __iadd__(self, other):
		for midx1, col in other.cols.iteritems():
			for midx2, val in col.coeffs.iteritems():
				try:
					self[midx1][midx2] += val
				except KeyError, e:
					# didn't exist yet
					self[midx1][midx2] = val
		self.d0 = max(self.d0, other.d0)
		self.d1 = max(self.d1, other.d1)
		return self

	def __sub__(self, other):
		""" Difference of two linear transformations """
		if not self.n == other.n:
			raise TypeError('Dimension mismatch')
		ret = self
		for midx1, col in other.cols.iteritems():
			for midx2, val in col.coeffs.iteritems():
				try:
					ret[midx1][midx2] -= val
				except KeyError, e:
					# didn't exist yet
					ret[midx1][midx2] = -val
		ret.d0 = max(self.d0, other.d0)
		ret.d1 = max(self.d1, other.d1)
		return ret

	def __mul__(self, other):
		""" Product of two linear transformations """
		if not self.n == other.n:
			raise TypeError('Dimension mismatch')
		ret = PolyLinTrans(self.n)
		for midx1, col in other.cols.iteritems():
			for midxk, val1 in other[midx1].coeffs.iteritems():
				for midx2, val2 in self[midxk].coeffs.iteritems():
					ret[midx1][midx2] += val1 * val2

		ret.d0 = other.d0
		ret.d1 = self.d1
		return ret

	def __neg__(self):
		ret = self
		for midx1, col in ret.cols.iteritems():
			for midx2, val in col.coeffs.iteritems():
				col.coeffs[midx2] = -val
		return ret

	def purge(self):
		""" Remove zeros in representation """
		for midx1, col in self.cols.iteritems():
			remove = [k for k,v in col.coeffs.iteritems() if v == 0.]
			for k in remove: del col.coeffs[k]
		remove = [k for k,v in self.cols.iteritems() if len(v.coeffs) == 0]
		for k in remove: del self.cols[k]

	def updated(self):
		self.d0 = 0
		self.d1 = 0
		for key1, col in self.cols.iteritems():
			for key2, val in col.coeffs.iteritems():
				self.d1 = max(self.d1, sum(key2))
			self.d0 = max(self.d0, sum(key1))

	def to_sparse(self):
		""" 
			Return a grlex-ordered representation A of the transformation.

			That is, if p is a grlex coefficient vector, 
				A p 
			is a grlex coefficient vector of the transformed polynomial.
		"""
		i = []
		j = []
		v = []
		nrow = count_monomials_leq(self.n, self.d1)
		ncol = count_monomials_leq(self.n, self.d0)
		for key1, col in self.cols.iteritems():
			midx1 = grlex_to_index(key1)
			for key2, val in col.coeffs.iteritems():
				i.append(grlex_to_index(key2))
				j.append(midx1)
				v.append(val)
		return nrow, ncol, i, j, v

	def to_sparse_matrix(self):
		""" Return a representation A of the transformation from a vector
			representing a symmetric matrix.

			That is, if p = vec(S) is a symmetric matrix vector, 
				A p 
			is a grlex coefficient vector of the transformed polynomial Trans( x^T S x ).
		"""
		half_deg = int(ceil(float(self.d0)/2))
		num_mon = count_monomials_leq(self.n, half_deg)
		len_vec = num_mon*(num_mon+1)/2

		coefs, exps = vec_to_grlex(len_vec, self.n)
		
		i = []
		j = []
		v = []

		for k, (coef,midx_mid) in enumerate(zip(coefs, exps)):
			for midx_f, val in self[midx_mid].coeffs.iteritems():
				i.append( grlex_to_index(midx_f) )
				j.append( k )
				v.append( val * coef )

		nrow = count_monomials_leq(self.n, self.d1)
		ncol = len_vec
		return nrow, ncol, i, j, v
