from sympy import Matrix

import unittest

from polylintrans import *
from density_sdsos import Lf

class PolyLinTransTests(unittest.TestCase):

	def test_index_to_multiindex(self):
		self.assertEquals(index_to_multiindex(29, 3), (2,0,2))
		self.assertEquals(index_to_multiindex(0, 3), (0,0,0))
		self.assertEquals(index_to_multiindex(9, 3), (2,0,0))

	def test_index_to_multiindex_to_index(self):
		for i in range(100):
			self.assertEquals(multiindex_to_index(index_to_multiindex(i, 3)), i)

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

	def test_L3(self):
		L = Lf(10, [(   ((0,3,0), 1),  ), (  ((0, 1, 4), 1),   )] )
		self.assertEqual(L.d0, 10)
		self.assertEqual(L.d1,  14)

	def test_lvar(self):
		L = PolyLinTrans.elvar(2,2,0,0)
		self.assertEqual(L[0,0][0,0], 1)
		self.assertEqual(L[0,1][0,1], 1)
		self.assertEqual(L[0,2][0,2], 1)
		self.assertEqual(L[1,0][1,0], 0)

	def test_lvar2(self):
		L = PolyLinTrans.elvar(2,2,0,1)
		self.assertEqual(L[0,0][0,0], 1)
		self.assertEqual(L[0,1][0,1], 1)
		self.assertEqual(L[0,2][0,2], 1)
		self.assertEqual(L[1,0][1,0], 0)
		self.assertEqual(L[1,0][0,0], 1)

	def test_mulpol(self):
		L = PolyLinTrans.mul_pol(2, 3, ( ((2,0), 1,), ((0,2), -1), ((1,0), 3) ) )
		self.assertEqual(L.d0, 3)
		self.assertEqual(L.d1, 5)
		self.assertEqual(L[0,0][1,0], 3)

	def test_sparse_matrix(self):
		L = PolyLinTrans(2)
		L[1,0][0,1] = 3
		L.updated()
		nrow, ncol, i, j, val = L.to_sparse_matrix()
		self.assertEquals(ncol, 6)
		self.assertEquals(nrow, 3)
		self.assertEquals(i[0], 1)
		self.assertEquals(j[0], 2)
		self.assertEquals(val[0], 6)

	def test_sparse_matrix2(self):
		L = PolyLinTrans.eye(2,2)
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

	def test_int(self):
		L = PolyLinTrans.int(2,2,0)
		self.assertEquals(L[1,1][2,1], 1./2)
		self.assertEquals(L[1,0][2,0], 1./2)
		self.assertEquals(L[2,0][1,0], 0)
		self.assertEquals(L[0,1][1,1], 1)
		self.assertEquals(L[2,0][3,0], 1./3) # above degree

	def test_integrate(self):
		L = PolyLinTrans.integrate(3,3,[0,1],[[0,1],[0,1]])
		self.assertEquals(L[0,0,2][0,0,2], 1)
		self.assertEquals(L[1,2,0][0,0,0], 1./6)
		self.assertEquals(L[1,1,1][0,0,1], 1./4)

	def test_integrate2(self):
		L = PolyLinTrans.integrate(3,3,[0],[[-5,1]])
		self.assertEquals(L[0,2,1][0,2,1], 6)
		self.assertEquals(L[1,2,0][0,2,0], -12)

	def test_evallintrans(self):
		L = PolyLinTrans.eye(2,3)
		p2 = eval_Lintrans(2, L, [1,2,3])
		self.assertEquals(p2[0:3], [1,2,3])
		L = PolyLinTrans.diff(2,3,1)
		p2 = eval_Lintrans(2, L, [1,2,3,4,5])
		self.assertEquals(p2[0:3], [2,8,5])


if __name__ == '__main__':
	unittest.main()