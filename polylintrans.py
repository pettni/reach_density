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

def multi_grlex_iter(midx, groups, degrees):
	'''
	Create an iterator that produces ordered exponents, starting
	with the multiindex 'midx', such that total degree of the variables
	in groups[i] does not exceed degrees[i].

	The ordering is based on grlex groupwise, but does not result
	in an overall grlex ordering.

	Example: The iterator grlex_iter( (0,0), [ [0], [1] ], [2], [1] ) 
			 produces the sequence
		(0,0) (1,0) (2,0) (0,1) (1,1) (2,1)
	'''

	# make sure 'groups' is a valid partition
	assert(set([dim for group in groups for dim in group]) == set(range(len(midx))))
	assert(len(groups) == len(degrees))

	# starting multiindices
	start_midx = [ tuple([ midx[dim] for dim in group ]) for group in groups ]

	iterators = [grlex_iter(m0, deg) for m0, deg in zip(start_midx, degrees) ]
	mons = [next(iterator) for iterator in iterators]

	ret = list(midx)
	while True:

		yield tuple(ret)

		for ind in range(len(degrees)):
			# find first that is not at max
			try:
				mons[ind] = next(iterators[ind])
				break
			except StopIteration:

				if ind == len(degrees) - 1:
					raise StopIteration
				
				# was at max, reset it
				iterators[ind] = grlex_iter(start_midx[ind], degrees[ind])
				mons[ind] = next(iterators[ind])

		# Fill out return tuple
		for group_nr, group in enumerate(groups):
			for pos_nr, pos in enumerate(group):
				ret[pos] = mons[group_nr][pos_nr]

def grlex_iter(midx, deg = -2):
	'''
	Create an iterator that produces ordered grlex exponents, starting
	with the multiindex 'midx'. The iterator stops when the total degree 
	reaches 'deg'+1. (default: never stop)

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

		if sum(midx) == deg + 1:
			raise StopIteration

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

	def __init__(self, n0, n1):
		self.n0 = n0 		# number of variables
		self.n1 = n1 		# final number of variables
		self.d0 = 0   	# initial degree
		self.d1 = 0		# final degree
		self.cols = {}

	def rows(self):
		return count_monomials_leq(self.n1, self.d1)

	def cols(self):
		return count_monomials_leq(self.n0, self.d0)

	def eye(n0, n1, d):
		'''
		Identity transformation
		'''
		if n1 < n0:
			raise InputError('eye requires n1 >= n0')
		p = PolyLinTrans(n0, n1)
		p.d0 = d
		for idx in grlex_iter((0,) *  n0, d):
			idx_mod = idx + (0,) * (n1-n0)
			p[idx][idx_mod] = 1.
		p.d1 = d
		return p
	eye = staticmethod(eye)

	def diff(n, d, xi):
		'''
		Differentiation transformation w.r.t variable xi
		'''
		p = PolyLinTrans(n,n)
		p.d0 = d
		for idx in grlex_iter((0,)*n, d):
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
		p = PolyLinTrans(n,n)
		p.d0 = d
		p.d1 = d+1
		for idx in grlex_iter((0,)*n, d):
			k = idx[xi]
			new_idx = tuple([(idx[i] if i != xi else idx[i]+1) for i in range(n)])
			p[idx][new_idx] = 1./(float(k)+1)
		return p
	int = staticmethod(int)

	def elvar(n, d, xi, val):
		'''
		Transformation resulting from setting xi[i] = val[i] (new polynomial has less variables)
		'''
		if isinstance(xi, list):
			xi, val = [list(x) for x in zip(*sorted(zip(xi, val), key=lambda pair: pair[0]))]
			if len(xi) > 1:
				return PolyLinTrans.elvar(n-1,d,xi[:-1], val[:-1]) * PolyLinTrans.elvar(n,d,xi[-1], val[-1])
			else:
				return PolyLinTrans.elvar(n,d,xi[0], val[0])
		p = PolyLinTrans(n,n-1)
		for idx in grlex_iter((0,)*n, d):
			new_idx = [idx[i] for i in range(len(idx)) if i != xi]
			p[idx][tuple(new_idx)] += val**idx[xi]
		p.updated()
		return p
	elvar = staticmethod(elvar)

	def mul_pol(n, d, poly):
		'''
		Transformation representing multiplication of a degree d polynomial with a
		polynomial poly represented as ( ( midx1, cf1 ), (midx2, cf2), ... )
		'''
		p = PolyLinTrans(n,n)
		maxdeg = 0
		for midx, cf in poly:
			for idx in grlex_iter((0,)*n, d):
				new_idx = tuple([idx[i] + midx[i] for i in range(n)])
				p[idx][new_idx] = float(cf)
			maxdeg = max([maxdeg, sum(midx)])
		p.d0 = d
		p.d1 = d + maxdeg
		return p
	mul_pol = staticmethod(mul_pol)

	def integrate(n, d, dims, boxes):
		'''
		Transformation representing integration over variables in 'dims'
		over a hyperbox 'box'

		Ex: The mapping p(x,y,z) |--> q(x,z) = \int_0^1 p(x,y,z) dy 
			is obtained by mon(q) = A_int * mon(p) for 
			>> A_int = integrate(3,2,[1],[[0,1]])
		'''
		dim_box = sorted(zip(dims, boxes), key = lambda obj : -obj[0])

		p = PolyLinTrans.eye(n,n,d)  # start with right-hand identity

		for (dim, box) in dim_box:
			int_trans = PolyLinTrans.int(p.n1,d,dim)  # p.n1 -> p.n1
			upper_trans = PolyLinTrans.elvar(p.n1,d+1,dim,box[1]) # p.n1 -> p.n1-1
			lower_trans = PolyLinTrans.elvar(p.n1,d+1,dim,box[0]) # p.n1 -> p.n1-1
			p = (upper_trans - lower_trans) * int_trans * p

		# for (i, xi) in enumerate(dims):
		# 	int_trans = PolyLinTrans.int(n,d,xi)
		# 	upper_trans = PolyLinTrans.elvar(n,d+1,xi,box[i][1])
		# 	lower_trans = PolyLinTrans.elvar(n,d+1,xi,box[i][0])
		# 	p = (upper_trans - lower_trans) * int_trans * p
		p.updated() # degrees become misleading here..
		return p
	integrate = staticmethod(integrate)

	def __getitem__(self, midx):
		if len(midx) != self.n0:
			raise TypeError('Multiindex does not match polynomial dimension')
		try:
			return self.cols[midx]
		except KeyError, e:
			self.cols[midx] = PolyLinTransRow(self.n1)
			return self.cols[midx]

	def __str__(self):
		ret = 'Transformation from n=%d, d=%d to n=%d, d=%d : \n' % (self.n0, self.d0, self.n1, self.d1)
		for key1, col in self.cols.iteritems():
			for key2, val in col.coeffs.iteritems():
				ret += str(key1) + ' --> ' + str(key2) + ' : ' + str(val) + '\n'
		return ret

	def __add__(self, other):
		""" Sum of two linear transformations """
		if not self.n0 == other.n0 and self.n1 == other.n1:
			raise TypeError('Dimension mismatch')
		ret = PolyLinTrans(self.n0, self.n1)
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
		if not self.n0 == other.n0 and self.n1 == other.n1:
			raise TypeError('Dimension mismatch')
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
		if not self.n0 == other.n0 and self.n1 == other.n1:
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
		""" Product of two linear transformations (other is the right one) """
		if not self.n0 == other.n1:
			raise TypeError('Dimension mismatch')
		ret = PolyLinTrans(other.n0, self.n1)
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
		nrow = count_monomials_leq(self.n1, self.d1)
		ncol = count_monomials_leq(self.n0, self.d0)
		for midx1, col in self.cols.iteritems():
			idx1 = grlex_to_index(midx1)
			for midx2, val in col.coeffs.iteritems():
				i.append(grlex_to_index(midx2))
				j.append(idx1)
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
		num_mon = count_monomials_leq(self.n0, half_deg)
		len_vec = num_mon*(num_mon+1)/2

		coefs, exps = vec_to_grlex(len_vec, self.n0)
		
		i = []
		j = []
		v = []

		for k, (coef,midx_mid) in enumerate(zip(coefs, exps)):
			for midx_f, val in self[midx_mid].coeffs.iteritems():
				i.append( grlex_to_index(midx_f) )
				j.append( k )
				v.append( val * coef )

		nrow = count_monomials_leq(self.n1, self.d1)
		ncol = len_vec
		return nrow, ncol, i, j, v
