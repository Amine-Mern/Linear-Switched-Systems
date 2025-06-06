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
        
    def test_reach_reduction(self):
        x0 = np.array([1.0, 0.0])
        Reach_mat, r, Ar, Br, Cr, x0r = self.sys.reach_reduction(x0)
        
        self.assertEqual(Reach_mat.shape[0], 2)
        self.assertEqual(Reach_mat.shape[1], r)
        
        ortho_test = Reach_mat.T @ Reach_mat
        np.testing.assert_array_almost_equal(ortho_test, np.eye(r), decimal=6,
                  err_msg="The columns of Reach_mat are not orthonormal")
        
        self.assertEqual(Ar.shape, (r, r, 2), "Ar shape incorrect")
        self.assertEqual(Br.shape, (r, 1, 2), "Br shape incorrect")
        self.assertEqual(Cr.shape, (1, r), "Cr shape incorrect")
        self.assertEqual(x0r.shape, (r, 1), "x0r shape incorrect")
        
        self.assertLessEqual(r, 2)
        
        
    def test_observability_reduction(self):
        x0 = np.array([0.0, 0.0])
        Obs_mat, r, Ao, Bo, Co, x0o = self.sys.obs_reduction(x0)
        
        ortho_test = Obs_mat.T @ Obs_mat
        np.testing.assert_array_almost_equal(ortho_test, np.eye(r), decimal=6,
              err_msg="The columns of Obs_mat are not orthonormal")
        
        self.assertEqual(Ao.shape, (r, r, 2), "Ao shape incorrect")
        self.assertEqual(Bo.shape, (r, 1, 2), "Bo shape incorrect")
        self.assertEqual(Co.shape, (1, r), "Co shape incorrect")
        self.assertEqual(x0o.shape, (r, 1), "x0o shape incorrect")
        
        self.assertLessEqual(r, 2)
        