import unittest
import numpy as np

from src.lpv.hoKalmanIdentifier import HoKalmanIdentifier
from src.lpv.mode.strategy_psi import strategy_psi
from src.lpv.mode.strategy_Myu import strategy_Myu

class hoKalmanIdentifierTest(unittest.TestCase):

    def setUp(self):
        # we define psig here as defined in the original matlab code to check if the functions are correct
        self.psig = np.array([[1],[1]])

        self.A = np.zeros((2, 3, 3))

        self.A[0, :, :] = np.array([
            [0.1039,0.0255,0.5598],
            [0.4338,0.0067,0.0078],
            [0.3435,0.0412,0.0776]
        ])

        
        self.A[1,:,:] = np.array([
            [0.1834,0.2456,0.0511],
            [0.0572,0.2445,0.0642],
            [0.1395,0.6413,0.5598]
        ])

        self.B = np.zeros((2, 3, 1))

        self.B[0,:,:] = np.array([[1.6143],[5.9383],[7.3671]])

        self.B[1,:,:] = np.array([[6.0624],[4.9800],[3.1372]])

        self.C = np.array([[0.1144, 0.7623, 0.0020]])
        
        self.D = np.eye(1)

        self.K = np.zeros((2, 3, 1))

        self.K[0, :, :] = np.array([[0.4942],[0.2827],[0.8098]]) 

        self.K[1, :, :] = np.array([[0.6215],[0.1561],[0.7780]])
        
        self.Q = np.zeros((2, 1, 1))

        self.Q[0, :, :] = np.array([[0.9707]])
        self.Q[1, :, :] = np.array([[0.9707]])

        self.beta = np.array([
        [1,  [], 0],
        [0,  1,  0],
        [0,  0,  0]
        ], dtype=object)

        self.alpha = np.array([
        [0, [0,0], 0],
        [0, 0, 0],
        [0, [], 0]
        ], dtype=object)

        self.base = (self.alpha,self.beta)
        
        #hokalmanIdentifier (shortned to HKI)
        self.HKI = HoKalmanIdentifier(self.A,self.B,self.C,self.D,self.K,self.Q,self.base)

    def test_switch_modes(self):
        
        self.assertTrue(isinstance(self.HKI.mode,strategy_psi))

        self.HKI.switchMode()

        self.assertTrue(isinstance(self.HKI.mode,strategy_Myu))


    def test_compute_Hab(self):
        # Matrix simulated in MATLAB
        Expected_Hab = np.array([
            [1.06168774375513, 1.77067424830340, 0.561253977983782],
            [2.34126034403033, 1.31984524333465, 1.62763156384145],
            [4.49596505087895, 1.79758044633223, 1.11881609865121]
        ])

        P = self.HKI.as_intern_sys.compute_Pi(self.psig,self.HKI.Q)
        G = self.HKI.as_intern_sys.compute_Gi(self.psig,self.HKI.Q,P)

        Hab_Psi = self.HKI.compute_Hab(self.psig,G)
        
        self.HKI.switchMode()

        Hab_Myu = self.HKI.compute_Hab(self.psig,G)

        self.assertTrue(np.allclose(np.round(Expected_Hab,4),Hab_Myu,rtol=1e-3,atol=1e-4))

        self.assertTrue(np.allclose(np.round(Expected_Hab,4),Hab_Psi,rtol = 1e-3,atol = 1e-4))

    def test_compute_Habk(self):
        Expected_Habk = np.zeros((2, 3, 3))
        
        # Matrix simulated in MATLAB
        
        Expected_Habk[0, :, :] = [
            [0.690573580576835, 0.603938372071722, 0.430853159159672],
            [1.06168774375513,  1.77067424830340,  0.561253977983782],
            [2.34126034403033,  1.31984524333465,  1.62763156384145]
        ]

        Expected_Habk[1, :, :] = [
            [1.32390840523592,  1.31270950149550,  0.449849373079192],
            [1.27720790168026,  0.888219064834115, 0.508406448000943],
            [1.64279877005683,  1.02940234726401,  0.536759059929827]
        ]

        P = self.HKI.as_intern_sys.compute_Pi(self.psig,self.HKI.Q)
        G = self.HKI.as_intern_sys.compute_Gi(self.psig,self.HKI.Q,P)

        Habk_Psi = self.HKI.compute_Habk(self.psig,G)

        self.HKI.switchMode()   

        Habk_Myu = self.HKI.compute_Habk(self.psig,G)
        
        self.assertTrue(np.allclose(np.round(Expected_Habk,4),Habk_Psi,rtol=1e-3,atol=1e-4))

        self.assertTrue(np.allclose(np.round(Expected_Habk,4),Habk_Myu,rtol=1e-3,atol=1e-4))

    def test_compute_Hak(self):
        Expected_Hak = np.zeros((2, 3, 1))

        # Matrix simulated in MATLAB
       
        Expected_Hak[0, :, :] = [
            [1.62763156384145],
            [1.11881609865121],
            [4.72600119411926]
        ]

        Expected_Hak[1, :, :] = [
            [1.06168774375513],
            [2.34126034403033],
            [4.49596505087895]
        ]

        P = self.HKI.as_intern_sys.compute_Pi(self.psig,self.HKI.Q)
        G = self.HKI.as_intern_sys.compute_Gi(self.psig,self.HKI.Q,P)

        Hak_Psi = self.HKI.compute_Hak(self.psig,G)

        self.HKI.switchMode()

        Hak_Myu = self.HKI.compute_Hak(self.psig,G)

        self.assertTrue(np.allclose(np.round(Expected_Hak,4),Hak_Psi,rtol=1e-3,atol=1e-4))

        self.assertTrue(np.allclose(np.round(Expected_Hak,4),Hak_Myu,rtol=1e-3,atol=1e-4))
    
    def test_compute_Hkb(self):
        # Matrix simulated in MATLAB
        
        Expected_Hkb = np.array([4.49596505087895, 1.79758044633223, 1.11881609865121])

        P = self.HKI.as_intern_sys.compute_Pi(self.psig,self.HKI.Q)
        G = self.HKI.as_intern_sys.compute_Gi(self.psig,self.HKI.Q,P)

        Hkb_Psi = self.HKI.compute_Hkb(self.psig,G)

        self.HKI.switchMode()

        Hkb_Myu = self.HKI.compute_Hkb(self.psig,G)

        print("Hkb")
        
        print(Hkb_Psi.shape)
        print(Hkb_Psi)

        print("---")
        print(Expected_Hkb)

        self.assertTrue(np.allclose(np.round(Expected_Hkb,4),Hkb_Psi,rtol=1e-3,atol=1e-4))

        self.assertTrue(np.allclose(np.round(Expected_Hkb,4),Hkb_Myu,rtol=1e-3,atol=1e-4))
        
    def test_identify(self):

        expected_A = np.zeros((2, 3, 3))
        
        expected_A[0, :, :] = [
            [0.5128, 0.0395, 0.3983],
            [0.1475, -0.0127, 0.1034],
            [-0.2049, 1.0414, -0.3120]
        ]

        expected_A[1, :, :] = [
            [0.0615, -0.0966, 0.0005],
            [0.6597, 0.7837, 0.2086],
            [0.1613, 0.0492, 0.1424]
            ]


        expected_B = np.zeros((2, 3, 1))

        expected_B[0, :, :] = [
            [1.1481],
            [0.7220],
            [-1.5496]
        ]

        expected_B[1, :, :] = [
            [1],
            [0],
            [0]
        ]

        expected_C = np.array([4.4960,1.7976,1.1188])

        Hab,Habk,Hak,Hkb = self.HKI.TrueHoKalmanBase(self.psig)

        dLPV_sys = self.HKI.identify(Hab,Habk,Hak,Hkb)

        self.assertTrue(np.allclose(expected_A,dLPV_sys.A,rtol=1e-3,atol=1e-4))
        self.assertTrue(np.allclose(expected_B,dLPV_sys.B,rtol=1e-3,atol=1e-4))
        self.assertTrue(np.allclose(expected_C,dLPV_sys.C,rtol=1e-3,atol=1e-4))
        self.assertTrue(np.allclose(self.D,dLPV_sys.D,rtol=1e-3,atol=1e-4))

        print("---------------")
        print(expected_C)

        print("---------------")


        print(dLPV_sys.C.shape)
        print(dLPV_sys.C)




        
