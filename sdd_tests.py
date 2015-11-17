import unittest

import sympy as sp
import numpy as np

from sympy.abc import x,y,z,t
from sympy.polys.monomials import itermonomials, monomial_count

from sdd import _k_to_ij, _ij_to_k, _sdd_index, is_dd, is_sdd

class sddTest(unittest.TestCase):

	def test_is_dd(self):
		self.assertTrue(is_dd(np.array([[1, 0],[0, 1]] )))
		self.assertTrue(is_dd(np.array([[1, 1],[1, 1]] )))
		self.assertFalse(is_dd(np.array([[1, 1.01],[1.01, 1]] )))
		self.assertFalse(is_dd(np.array([[1, 1.01,0],[1.01, 1, 0], [0,0,1]] )))
		self.assertFalse(is_dd(np.array([[1, 0.3,-0.69],[0.3, 1, 0.69], [-0.69,0.69,1]] )))
		self.assertTrue(is_dd(np.array([[1, 0.3,-0.69],[0.3, 1, 0.69], [-0.69,0.69,1.5]] )))
		self.assertTrue(is_sdd(np.array([[1, 0.3,-0.69],[0.3, 1, 0.69], [-0.69,0.69,1.5]] )))

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