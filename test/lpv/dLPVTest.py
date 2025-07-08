import unittest
import numpy as np
from src.lpv.dLPV import dLPV
from src.lpv.asLPV import asLPV

class dLPVTest(unittest.TestCase):
    def setUp(self):
        A = np.zeros((2, 2, 2))
        B = np.zeros((2, 2, 1))
        C = np.array([[1.0, 0.0]])
        D = np.array([[0.0]])

        A[0, :, :] = np.array([[0.8, 0.1],
                                [0.0, 0.5]])
        A[1, :, :] = np.array([[0.5, 0.2],
                                [0.1, 0.6]])
        B[0, :, :] = np.array([[1.0],
                                [0.0]])
        B[1, :, :] = np.array([[0.5],
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
        
        self.assertEqual(Ar.shape, (2, r, r), "Ar shape incorrect")
        self.assertEqual(Br.shape, (2, r, 1), "Br shape incorrect")
        self.assertEqual(Cr.shape, (1, r), "Cr shape incorrect")
        self.assertEqual(x0r.shape, (r, 1), "x0r shape incorrect")
        
        self.assertLessEqual(r, 2)
        
        
    def test_observability_reduction(self):
        x0 = np.array([0.0, 0.0])
        Obs_mat, r, Ao, Bo, Co, x0o = self.sys.obs_reduction(x0)
        
        ortho_test = Obs_mat.T @ Obs_mat
        np.testing.assert_array_almost_equal(ortho_test, np.eye(r), decimal=6,
              err_msg="The columns of Obs_mat are not orthonormal")
        
        self.assertEqual(Ao.shape, (2, r, r), "Ao shape incorrect")
        self.assertEqual(Bo.shape, (2, r, 1), "Bo shape incorrect")
        self.assertEqual(Co.shape, (1, r), "Co shape incorrect")
        self.assertEqual(x0o.shape, (r, 1), "x0o shape incorrect")
        
        self.assertLessEqual(r, 2)
        
        
    def test_minimal_realization(self):
        x0 = np.array([1.0, 0.0])
        minimal_sys, x0m = self.sys.minimize(x0)
            
        r = minimal_sys.A.shape[0]
        self.assertIsInstance(minimal_sys, dLPV)
        self.assertEqual(minimal_sys.A.shape, (self.sys.np, r, r), "A shape is incorrect")
        self.assertEqual(minimal_sys.B.shape, (self.sys.np, r, self.sys.nu), "B shape is incorrect")
        self.assertEqual(minimal_sys.C.shape, (self.sys.ny, r), "C shape is incorrect")
        self.assertEqual(minimal_sys.D.shape, (self.sys.ny, self.sys.nu), "D shape is incorrect")
        self.assertEqual(x0m.shape, (r, 1), "x0 shape is incorrect")
        self.assertLessEqual(r, self.sys.nx)
        
    def test_isomorphic_systems(self):
        A2 = np.zeros((2,2,2))
        B2 = np.zeros((2,2,1))
        C2 = np.zeros((1,2))
        D2 = np.zeros((1,1))
        
        T = np.array([[1,2], [0,1]])
        T_inv = np.linalg.inv(T)
        for i in range(2):
            A2[i,:,:] = T @ self.sys.A[i,:,:] @ T_inv
            B2[i,:,:] = T @ self.sys.B[i,:,:]
        C2 = self.sys.C @ T_inv
        D2 = self.sys.D
        sys2 = dLPV (A2, C2, B2, D2)
        x0 = np.array([0.0, 0.0])
        x02 = np.array([0.0, 0.0])
        self.assertTrue(self.sys.isIsomorphic(sys2, x0,x02))
        
    def test_non_isomorphic_systems(self):
        A2 = np.zeros((2,2,2))
        B2 = np.zeros((2,2,1))
        C2 = np.zeros((1,2))
        D2 = np.zeros((1,1))
        sys2 = dLPV (A2, C2, B2, D2)
        x0 = np.array([0.0, 0.0])
        x02 = np.array([0.0, 0.0])
        self.assertFalse(self.sys.isIsomorphic(sys2, x0,x02))
        
    def test_recursion(self):
        nx = 2
        ny = 1
        np_ = 2
        A = np.zeros((np_, nx, nx))
        B = np.zeros((np_, nx, ny))
        C = np.array([[1.0, 0.0]])

        A[0, :, :] = np.array([[0.9, 0.1],
                              [0.0, 0.8]])
        A[1, :, :] = np.array([[0.7, 0.2],
                              [0.1, 0.6]])
        B[0, :, :] = np.array([[0.1],
                              [0.0]])
        B[1, :, :] = np.array([[0.05],
                              [0.02]])
        T_sig = np.zeros((np_, ny, ny))
        T_sig[0, :, :] = np.array([[10.0]])
        T_sig[1, :, :] = np.array([[5.0]])

        psig = np.array([[0.6], [0.4]])

        sys = dLPV(A, C, B, None)
        sys.nx = nx
        sys.ny = ny
        sys.np = np_

        P, Q, K = sys.Recursion(T_sig, psig)
        
        self.assertEqual(P.shape, (self.sys.np, self.sys.nx, self.sys.nx))
        self.assertEqual(Q.shape, (self.sys.np, self.sys.ny, self.sys.ny))
        self.assertEqual(K.shape, (self.sys.np, self.sys.nx, self.sys.ny))
        
    def test_convert_to_asLPV(self):
        nx = 2
        ny = 1
        np_ = 2
        A = np.zeros((np_, nx, nx))
        B = np.zeros((np_, nx, ny))
        C = np.array([[1.0, 0.0]])

        A[0, :, :] = np.array([[0.9, 0.1],
                              [0.0, 0.8]])
        A[1, :, :] = np.array([[0.7, 0.2],
                              [0.1, 0.6]])
        B[0, :, :] = np.array([[0.1],
                              [0.0]])
        B[1, :, :] = np.array([[0.05],
                              [0.02]])
        T_sig = np.zeros((np_, ny, ny))
        T_sig[0, :, :] = np.array([[10.0]])
        T_sig[1, :, :] = np.array([[5.0]])

        psig = np.array([[0.6], [0.4]])

        self.sys = dLPV(A, C, B, None)
        self.sys.nx = nx
        self.sys.ny = ny
        self.sys.np = np_
        as_sys, Qmin = self.sys.convert_to_asLPV(T_sig, psig)

        self.assertTrue(isinstance(as_sys, asLPV), "Returned system is not an instance of asLPV")
        self.assertEqual(Qmin.shape, (self.sys.np, self.sys.ny, self.sys.ny), "Qmin shape incorrect")
        self.assertEqual(as_sys.A.shape, self.sys.A.shape, "A shape mismatch")
        self.assertEqual(as_sys.K.shape, (self.sys.np, self.sys.nx, self.sys.ny), "K shape mismatch")
        self.assertEqual(as_sys.C.shape, self.sys.C.shape, "C shape mismatch")
        self.assertEqual(as_sys.F.shape, (self.sys.ny, self.sys.ny), "F shape mismatch")

