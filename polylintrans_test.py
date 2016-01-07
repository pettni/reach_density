import unittest

import numpy as np
import sympy as sp

import itertools

from sympy.polys.monomials import itermonomials
from sympy.polys.orderings import monomial_key
from sympy.abc import x, y, z

from polylintrans import *
from density_sdsos import Lf


class PolyLinTransTests(unittest.TestCase):

	def test_index_to_grlex(self):
		self.assertEquals(index_to_grlex(29, 3), (2,0,2))
		self.assertEquals(index_to_grlex(0, 3), (0,0,0))
		self.assertEquals(index_to_grlex(9, 3), (2,0,0))

	def test_index_to_grlex_to_index(self):
		for i in range(100):
			self.assertEquals(grlex_to_index(index_to_grlex(i, 3)), i)

	def test_L1(self):
		"""Test computing Lf for the special case
			rho = a + b t + c x,
			with vf = [3 x t].
			Lf is L rho = b + c*3xt
		"""
		L = Lf(1, [(((1,1), 3),)] )
		self.assertEqual(L[1,0][0,0], 1)
		self.assertEqual(L[0,1][1,1], 3)

	def test_L2(self):
		"""Test computing Lf for the special case
			rho = b + a0 t + a1 x1 + a2 x2 + a3 t^2 + a4 t x1 + a5 t x2
		 		+  a6 x1^2 + a7 x1 x2 + a8 x2^2,
			with vf = [1, -3 x1].
			Lf is L rho = (a0 + 2 a3 t + a4 x1 + a5 x2)
								+ (a1 + a4 t + 2 a6 x1 + a7 x2) * (1)
								+ (a2 + a5 t + a7 x1 + 2 a8 x2) * (-3 x1)
								= a0 + t [2 a3 + a4] + x1[ a4 + 2a6 - 3 a2 ] + x2 [a5 + a7] +
									x1 t [-3a5] + x1^2 [-3a7] + x1 x2 [-6 a8]
		"""
		L = Lf(2, [( ((0,0,0), 1), ), ( ((0, 1, 0), -3), )] )
		self.assertEqual(L[1,0,0][0,0,0], 1)
		self.assertEqual(L[2,0,0][1,0,0], 2)
		self.assertEqual(L[1,1,0][1,0,0], 1)
		self.assertEqual(L[0,0,2][0,1,1], -6)
		self.assertEqual(L[1,1,0][0,1,0], 1)

	def test_L4(self):
		"""Test computing Lf for the special case
			rho = b + a0 t + a1 x1 + a2 x2 + a3 t^2 + a4 t x1 + a5 t x2
		 		+  a6 x1^2 + a7 x1 x2 + a8 x2^2,
			with vf = [1, -3 x1].
			Lf is L rho = (a0 + 2 a3 t + a4 x1 + a5 x2)
								+ (a1 + a4 t + 2 a6 x1 + a7 x2) * (1)
								+ (a2 + a5 t + a7 x1 + 2 a8 x2) * (-3 x1)
								= a0 + t [2 a3 + a4] + x1[ a4 + 2a6 - 3 a2 ] + x2 [a5 + a7] +
									x1 t [-3a5] + x1^2 [-3a7] + x1 x2 [-6 a8]
		"""
		L = Lf(2, [( ((0,0,0,0), 1), ), ( ((0, 1, 0, 0), -3), ((0, 0, 0, 1), 1))] )
		self.assertEqual(L[1,0,0,0][0,0,0,0], 1)
		self.assertEqual(L[2,0,0,0][1,0,0,0], 2)
		self.assertEqual(L[1,1,0,0][1,0,0,0], 1)
		self.assertEqual(L[0,0,2,0][0,1,1,0], -6)
		self.assertEqual(L[1,1,0,0][0,1,0,0], 1)

		self.assertEqual(L[1,0,1,0][1,0,0,1], 1)
		self.assertEqual(L[0,0,1,1][0,0,0,2], 1)

	def test_L3(self):
		L = Lf(10, [(   ((0,3,0), 1),  ), (  ((0, 1, 4), 1),   )] )
		self.assertEqual(L.d0, 10)
		self.assertEqual(L.d1,  14)

	def test_lvar(self):
		L = PolyLinTrans.elvar(2,2,0,0)
		self.assertEqual(L[0,0][(0,)], 1)
		self.assertEqual(L[0,1][(1,)], 1)
		self.assertEqual(L[0,2][(2,)], 1)
		self.assertEqual(L[1,0][(0,)], 0)

	def test_multi_lvar(self):
		L = PolyLinTrans.elvar(3,3,[0, 1],[0,3])
		self.assertEqual(L[0,0,0][(0,)], 1)
		self.assertEqual(L[0,0,1][(1,)], 1)
		self.assertEqual(L[0,0,2][(2,)], 1)
		self.assertEqual(L[1,1,0][(0,)], 0)
		self.assertEqual(L[0,1,0][(0,)], 3)

	def test_multi_lvar2(self):
		L = PolyLinTrans.elvar(3,3,[2, 0],[3,1])
		self.assertEqual(L[0,0,0][(0,)], 1)
		self.assertEqual(L[0,0,1][(0,)], 3)
		self.assertEqual(L[0,0,2][(0,)], 9)
		self.assertEqual(L[2,0,0][(0,)], 1)

	def test_lvar2(self):
		L = PolyLinTrans.elvar(2,2,0,1)
		self.assertEqual(L[0,0][(0,)], 1)
		self.assertEqual(L[0,1][(1,)], 1)
		self.assertEqual(L[0,2][(2,)], 1)
		self.assertEqual(L[1,0][(0,)], 1)

	def test_mulpol(self):
		L = PolyLinTrans.mul_pol(2, 3, ( ((2,0), 1,), ((0,2), -1), ((1,0), 3) ) )
		self.assertEqual(L.d0, 3)
		self.assertEqual(L.d1, 5)
		self.assertEqual(L[0,0][1,0], 3)

	def test_sparse_matrix(self):
		L = PolyLinTrans(2,2)
		L[1,0][0,1] = 3
		L.updated()
		nrow, ncol, i, j, val = L.to_sparse_matrix()
		self.assertEquals(ncol, 6)
		self.assertEquals(nrow, 3)
		self.assertEquals(i[0], 1)
		self.assertEquals(j[0], 2)
		self.assertEquals(val[0], 6)

	def test_sparse_matrix2(self):
		L = PolyLinTrans.eye(2,2,2)
		nrow, ncol, indi, indj, val = L.to_sparse_matrix()
		for (i, idx) in enumerate(indi):
			self.assertEquals(idx, indj[i]) # diagonal
			if i in [0,3,5]:
				self.assertEquals(val[i], 1)
			else:
				self.assertEquals(val[i], 2)

	def test_diff(self):
		L = PolyLinTrans.diff(2,2,0)
		self.assertEquals(L[0,0][0,0], 0)
		self.assertEquals(L[1,0][0,0], 1)
		self.assertEquals(L[2,0][1,0], 2)
		self.assertEquals(L[4,0][3,0], 0) # above degree

	def test_mul(self):
		L = PolyLinTrans.eye(2,2,3)
		L2 = L * 3
		_,_,idxi,idxj,vals = L2.to_sparse()
		self.assertEquals(idxi, idxj)
		self.assertEquals(vals, [3] * 10)

	def test_int(self):
		L = PolyLinTrans.int(2,2,0)
		self.assertEquals(L[1,1][2,1], 1./2)
		self.assertEquals(L[1,0][2,0], 1./2)
		self.assertEquals(L[2,0][1,0], 0)
		self.assertEquals(L[0,1][1,1], 1)
		self.assertEquals(L[2,0][3,0], 1./3) # above degree

	def test_integrate(self):
		L = PolyLinTrans.integrate(3,3,[0,1],[[0,1],[0,1]])
		self.assertEquals(L[0,0,2][(2,)], 1)
		self.assertEquals(L[1,2,0][(0,)], 1./6)
		self.assertEquals(L[1,1,1][(1,)], 1./4)

	def test_integrate2(self):
		L = PolyLinTrans.integrate(3,3,[0],[[-5,1]])
		self.assertEquals(L[0,2,1][2,1], 6)
		self.assertEquals(L[1,2,0][2,0], -12)

	def test_evallintrans(self):
		L = PolyLinTrans.eye(2,2,3)
		p2 = eval_Lintrans(2, L, [1,2,3])
		self.assertEquals(p2[0:3], [1,2,3])
		L = PolyLinTrans.diff(2,3,1)
		p2 = eval_Lintrans(2, L, [1,2,3,4,5])
		self.assertEquals(p2[0:3], [2,8,5])

	def test_grlex_iter(self):
		iterator = grlex_iter( (0,0,0,0) )

		idx1 = next(iterator)

		for k in range(600):
			idx2 = next(iterator)
			self.assertTrue( grlex_comp( idx1, idx2 ) < 0 )
			idx1 = idx2

	def test_grlex_iter2(self):
		iterator = grlex_iter( (0,0,0,0), 4 )

		for k in range(count_monomials_leq(4,4)):
			next(iterator)
		
		try:
			next(iterator)
		except StopIteration:
			pass
		except Exception, e:
			self.fail('Unexpected exception thrown:')
		else:
			self.fail('ExpectedException not thrown')

	def test_grlex_iter_multi(self):
		m1 = 5
		m2 = 3
		m3 = 10
		iterator = multi_grlex_iter( (0,0,0), [[0], [1], [2]], [m1,m2,m3] )

		comb_iter = itertools.product( range(m1+1), range(m2+1), range(m3+1) )

		for k in range((m1+1) * (m2+1) * (m3+1)):
			asd =  next(iterator)
			if k == m1+1:
				self.assertEquals(asd, (0,1,0))

		self.assertEquals(asd , (m1,m2,m3))

		try:
			next(iterator)
		except StopIteration:
			pass
		except Exception, e:
			self.fail('Unexpected exception thrown:')
		else:
			self.fail('ExpectedException not thrown')

	def test_vec_to_grlex1(self):
		coef, exp = vec_to_grlex(10,3)
		self.assertEquals(exp, [(0,0,0), (0,0,1), (0,1,0), (1,0,0), (0,0,2), (0,1,1), (1,0,1), (0,2,0), (1,1,0), (2,0,0) ])
		self.assertEquals(coef, [1,2,2,2,1,2,2,1,2,1])

	def test_vec_to_grlex2(self):
		coef, exp = vec_to_grlex(10,2)
		self.assertEquals(exp, [(0,0), (0,1), (1,0), (0,2), (0,2), (1,1), (0,3), (2,0), (1,2), (0,4) ])
		self.assertEquals(coef, [1,2,2,2,1,2,2,1,2,1])

	def test_vec_to_grlex3(self):
		coef, exp = vec_to_grlex(28,1)
		self.assertEquals(exp, [tuple([i+j]) for j in range(7) for i in range(j,7)])
		self.assertEquals(coef, [1 if i==j else 2 for j in range(7) for i in range(j,7)])

	def test_vec_to_sparse_matrix1(self):
		poly_vec = [1,2,3,4,5,6,7,8,9,10]

		# represents 4 x 4 matrix :
		poly_mat = np.array([
					[1, 2, 3, 4],
		            [2, 5, 6, 7],
		            [3, 6, 8, 9],
		            [4, 7, 9, 10]])

		trans = vec_to_sparse_matrix(10,2)

		mon_vec = trans.dot(np.array(poly_vec))

		mon1 = sorted(itermonomials([x, y], 4), key=monomial_key('grlex', [x, y]))[0:4]
		mon2 = sorted(itermonomials([x, y], 4), key=monomial_key('grlex', [x, y]))[0:len(mon_vec)]

		self.assertEquals( sp.simplify(np.dot(mon_vec, mon2) - np.dot(np.dot( poly_mat, mon1 ), mon1 )), 0 )

	def test_vec_to_sparse_matrix2(self):
		poly_vec = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

		# represents 4 x 4 matrix :
		poly_mat = np.array([
					[1, 2, 3, 4, 5],
		            [2, 6, 7, 8, 9],
		            [3, 7, 10, 11, 12],
		            [4, 8, 11, 13, 14],
		            [5, 9, 12, 14, 15]])

		trans = vec_to_sparse_matrix(15,3)

		mon_vec = trans.dot(np.array(poly_vec))

		mon1 = sorted(itermonomials([x, y, z], 4), key=monomial_key('grlex', [x, y, z]))[0:5]
		mon2 = sorted(itermonomials([x, y, z], 4), key=monomial_key('grlex', [x, y, z]))[0:len(mon_vec)]

		self.assertEquals( sp.simplify(np.dot(mon_vec, mon2) - np.dot(np.dot( poly_mat, mon1 ), mon1 )), 0 )

if __name__ == '__main__':
	unittest.main()