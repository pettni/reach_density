import unittest

import sympy as sp
import numpy as np

from sympy.abc import x,y,z,t
from sympy.polys.monomials import itermonomials, monomial_count

from sdd import _k_to_ij, _ij_to_k, _sdd_index, get_symmetric_matrix, is_dd, is_sdd, degree, create_square_poly
from help_functions import extract_linear, gradient, compute_Lf, flatten

class sddTest(unittest.TestCase):

	def testdegree(self):
		self.assertEquals(degree(x, [x]), 1)
		self.assertEquals(degree(3*(x+y)**2, [x,y]), 2)
		self.assertEquals(degree((x+(-100)*y+z)**2, [x,y]), 2)
		self.assertEquals(degree((x+y+z)**2, [x,y,z]), 2)
		self.assertEquals(degree((x+y+z)**2 + 57*x*y, [x,y,z]), 2)
		self.assertEquals(degree((x**2)**2 + 57*x*y, [x,y,z]), 4)
		self.assertEquals(degree([x, y], [x,y,z]), 1)
		self.assertEquals(degree([x**2, y], [x,y,z]), 2)
		self.assertEquals(degree([x**2*z**2, y, z**3], [x,y,z]), 4)

	def testgradient(self):
		f = x**2*y
		gradf = gradient(f, [x,y])
		self.assertEquals(len(gradf), 2)
		self.assertEquals(gradf[0]- 2*x*y, 0)
		self.assertEquals(gradf[1]- x**2, 0)
		f = x**2*y*z
		gradf = gradient(f, [x,y,z])
		self.assertEquals(len(gradf), 3)
		self.assertEquals(gradf[0]- 2*x*y*z, 0)
		self.assertEquals(gradf[1]- x**2*z, 0)

	def test_computeLrho(self):
		f = x**2*y+t
		vf = [x, y]
		Lf = compute_Lf(f, vf, [t,x,y])
		self.assertEquals(Lf -  1- 2*x**2*y - x**2*y, 0)
		f = x
		vf = [x, y]
		Lf = compute_Lf(f, vf, [t,x,y])
		self.assertEquals(Lf-x, 0)

	def testflatten(self):
		self.assertEquals([1,2,3], flatten([[1,2],[3]]))

	def test_is_dd(self):
		self.assertTrue(is_dd(np.array([[1, 0],[0, 1]] )))
		self.assertTrue(is_dd(np.array([[1, 1],[1, 1]] )))
		self.assertFalse(is_dd(np.array([[1, 1.01],[1.01, 1]] )))
		self.assertFalse(is_dd(np.array([[1, 1.01,0],[1.01, 1, 0], [0,0,1]] )))
		self.assertFalse(is_dd(np.array([[1, 0.3,-0.69],[0.3, 1, 0.69], [-0.69,0.69,1]] )))
		self.assertTrue(is_dd(np.array([[1, 0.3,-0.69],[0.3, 1, 0.69], [-0.69,0.69,1.5]] )))
		self.assertTrue(is_sdd(np.array([[1, 0.3,-0.69],[0.3, 1, 0.69], [-0.69,0.69,1.5]] )))

	def test_square(self):
		for d in range(3):
			poly, A = create_square_poly(d, [x,y,z], 's')
			self.assertEquals(len(sp.Poly(poly, [x,y,z]).coeffs()), monomial_count(3, 2*d))

	def test_linear_trans1(self):
		A, b = extract_linear([x+y-3-4,y+z+2*y-1,z+x-1], [x, y, z])
		self.assertTrue(np.all(A == [[1., 1., 0], [0., 3., 1.], [1., 0., 1.]]))
		self.assertTrue(np.all(b == [7, 1, 1]))

	def test_linear_trans2(self):
		A, b = extract_linear([1.,0.], [x, y, z])
		self.assertTrue(np.all(A == [[0., 0., 0], [0., 0., 0.]]))
		self.assertTrue(np.all(b== [-1., 0.]))

	def test_linear_trans3(self):
		A, b = extract_linear([], [x, y, z])
		self.assertTrue(np.all(A == []))
		self.assertTrue(np.all(b == []))

	def test_linear_trans3(self):
		A, b = extract_linear([x-1,y-2,z-3], [x, y, z])
		self.assertTrue(np.all(A == np.eye(3)))
		self.assertTrue(np.all(b == [1,2,3]))

	def test_k_to_ij1(self):
		i, j = _k_to_ij(0, 10)
		self.assertEquals(i, 0)
		self.assertEquals(j, 0)
		i, j = _k_to_ij(9, 10)
		self.assertEquals(i, 3)
		self.assertEquals(j, 3)
		i, j = _k_to_ij(5, 10)
		self.assertEquals(i, 1)
		self.assertEquals(j, 2)

	def test_k_to_ij2(self):
		i, j = _k_to_ij(0, 1)
		self.assertEquals(i, 0)
		self.assertEquals(j, 0)

	def test_k_to_ij3(self):
		i, j = _k_to_ij(2, 6)
		self.assertEquals(i, 0)
		self.assertEquals(j, 2)
		i, j = _k_to_ij(3, 6)
		self.assertEquals(i, 1)
		self.assertEquals(j, 1)

	def test_ij_to_k(self):
		k = _ij_to_k(0,2,6) 
		self.assertEquals(k, 2)
		k = _ij_to_k(2,0,6) 
		self.assertEquals(k, 2)
		k = _ij_to_k(3,3,10) 
		self.assertEquals(k, 9)
		k = _ij_to_k(0,0,10) 
		self.assertEquals(k, 0)
		k = _ij_to_k(1,1,6) 
		self.assertEquals(k, 3)

	def test_symmetric_matrix(self):
		V = np.array([1,2,3,4,5,6])
		Q = get_symmetric_matrix(V)
		self.assertTrue(np.all(Q == np.array([[1,2,3], [2,4,5], [3, 5, 6]])))

	def test_symmetric_matrix2(self):
		self.assertRaises(TypeError, get_symmetric_matrix, np.array([1,2,3,4,5]))

	def test_sdd_index1(self):
		x11,x22,x12,y11,y22,y12,z11,z22,z12 = sp.symbols('x11,x22,x12,y11,y22,y12,z11,z22,z12')

		M = [[x11,x22,x12], [y11,y22,y12], [z11,z22,z12]]
	
		tt = [[0 for i in range(3)] for j in range(3)]
		for i in range(3):
			for j in range(3):
				for idx in _sdd_index(i,j,3):
					tt[i][j] = tt[i][j] + M[idx[0]][idx[1]]

		self.assertEquals(tt[0][0]-x11-y11, 0)
		self.assertEquals(tt[0][1]-x12, 0)
		self.assertEquals(tt[0][2]-y12, 0)
		self.assertEquals(tt[1][1]-x22-z11, 0)
		self.assertEquals(tt[1][2]-z12, 0)
		self.assertEquals(tt[2][2]-z22-y22, 0)

	def test_sdd_index2(self):
		x11,x22,x12 = sp.symbols('x11,x22,x12')

		M = [[x11,x22,x12]]
	
		tt = [[0 for i in range(2)] for j in range(2)]
		for i in range(2):
			for j in range(2):
				for idx in _sdd_index(i,j,2):
					tt[i][j] = tt[i][j] + M[idx[0]][idx[1]]
		self.assertEquals(tt[0][0]-x11, 0)
		self.assertEquals(tt[0][1]-x12, 0)
		self.assertEquals(tt[1][1]-x22, 0)

if __name__ == '__main__':
	unittest.main()