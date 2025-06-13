import unittest
import numpy as np
from src.lpv.LPV import LPV

class LPVTest(unittest.TestCase):

    def setUp(self):
        self.A1 = np.zeros((2, 2, 2))
        self.B1 = np.zeros((2, 1, 2))
        self.C1 = np.array([[1.0, 0.0]])
        self.D1 = np.array([[0.0]])

        self.A2 = np.zeros((2, 2, 2))
        self.B2 = np.zeros((2, 1, 2))

    def test_isEquivalent_true(self):
        
        self.A1[:, :, 0] = np.array([[0.8, 0.1],
                                [0.0, 0.5]])
        self.A1[:, :, 1] = np.array([[0.5, 0.2],
                                [0.1, 0.6]])
        self.B1[:, :, 0] = np.array([[1.0],
                                [0.0]])
        self.B1[:, :, 1] = np.array([[0.5],
                                [1.0]])
        
        
        T = np.array([[2.0, 0.0],[1.0, 1.0]])
        T_inv = np.linalg.inv(T)
        
        for i in range(2):
            self.A2[:, :, i] = T @ self.A1[:, :, i] @ T_inv
            self.B2[:, :, i] = T @ self.B1[:, :, i]
            
        C2 = self.C1 @ T_inv
        D2 = np.array([[0.0]])

        x01 = np.array([0.0, 0.0])
        x02 = np.array([0.0, 0.0])

        sys1 = LPV(self.A1, self.C1, self.B1, self.D1)
        sys2 = LPV(self.A2, C2, self.B2, D2)
        
        self.assertTrue(sys1.isEquivalentTo(sys2,x01,x02))
        
    def test_isEquivalent_false(self):
        self.A1[:, :, 0] = np.array([[0.9, 0.1],
                        [0.0, 0.5]])
        self.A1[:, :, 1] = np.array([[0.3, 0.0],
                                [0.0, 0.4]])

        self.B1[:, :, 0] = np.array([[1.0],
                                [0.0]])
        self.B1[:, :, 1] = np.array([[0.5],
                                [1.0]])

        self.A2[:, :, 0] = np.array([[0.6, 0.2],
                                [0.1, 0.5]])
        self.A2[:, :, 1] = np.array([[0.2, 0.1],
                                [0.1, 0.7]])

        self.B2[:, :, 0] = np.array([[0.8],
                                [0.1]])
        self.B2[:, :, 1] = np.array([[0.3],
                                [0.9]])

        C2 = self.C1.copy()
        D2 = self.D1.copy()

        x01 = np.array([0.0, 0.0])
        x02 = np.array([0.0, 0.0])

        sys1 = LPV(self.A1, self.C1, self.B1, self.D1)
        sys2 = LPV(self.A2, C2, self.B2, D2)
        self.assertFalse(sys1.isEquivalentTo(sys2,x01,x02))

            