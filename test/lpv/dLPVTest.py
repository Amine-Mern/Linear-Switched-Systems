import unittest
import numpy as np
from src.lpv.dLPV import dLPV

class dLPVTest(unittest.TestCase):
    def setUp(self):
        A = np.zeros((2, 2, 2))
        B = np.zeros((2, 1, 2))
        C = np.array([[1.0, 0.0]])
        D = np.array([[0.0]])

        A[:, :, 0] = np.array([[0.8, 0.1],
                                [0.0, 0.5]])
        A[:, :, 1] = np.array([[0.5, 0.2],
                                [0.1, 0.6]])
        B[:, :, 0] = np.array([[1.0],
                                [0.0]])
        B[:, :, 1] = np.array([[0.5],
                                [1.0]])
        self.sys = dLPV(A, C, B, D)
        
    def test_simulate_output(self):
        Ntot = 10
        u = np.ones((1, Ntot))
        p = np.ones((2, Ntot))
        y, yif, x = self.sys.simulate_y(u, p, Ntot)
        self.assertEqual(y.shape, (1, Ntot), "y has incorrect shape")
        self.assertEqual(yif.shape, (1, Ntot), "yif has incorrect shape")
        self.assertEqual(x.shape, (2, Ntot), "x has incorrect shape")
        np.testing.assert_array_almost_equal(y, yif)
        expected_x1 = np.array([1.5, 1.0])
        np.testing.assert_array_almost_equal(x[:,1], expected_x1, decimal=6)
        