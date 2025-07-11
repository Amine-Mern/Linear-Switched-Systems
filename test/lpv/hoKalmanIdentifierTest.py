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

        self.Q[0, :, :] = np.array([[0]])
        self.Q[1, :, :] = np.array([[0.9707]])

        self.Q_d = np.zeros((2, 1, 1))

        self.Q_d[0, :, :] = np.array([[1]])
        self.Q_d[1, :, :] = np.array([[1]])

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
        Expected_Hak_psi = np.zeros((2, 3, 1))

        # Matrix simulated in MATLAB
       
        Expected_Hak_psi[0, :, :] = [
            [1.6276],
            [1.1188],
            [4.7260]
        ]

        Expected_Hak_psi[1, :, :] = [
            [1.0616],
            [2.3412],
            [4.4959]
        ]

        Expected_Hak_myu = np.zeros((2, 3, 2))

        Expected_Hak_myu[0, :, :] = [
            [1.6276 , 0.1262],
            [1.1188 , 0.2632],
            [4.7260 , 0.3746]
        ]

        Expected_Hak_myu[1, : , :] = [
            [1.0617 , 0.4153],
            [2.3413 , 0.4507],
            [4.4960 , 0.4098]
        ]

        P = self.HKI.as_intern_sys.compute_Pi(self.psig,self.HKI.Q)
        G = self.HKI.as_intern_sys.compute_Gi(self.psig,self.HKI.Q,P)

        Hak_Psi = self.HKI.compute_Hak(self.psig,G)

        self.HKI.switchMode()

        Hak_Myu = self.HKI.compute_Hak(self.psig,G)

        self.assertTrue(np.allclose(Expected_Hak_psi,Hak_Psi,rtol=1e-3,atol=1e-4))

        self.assertTrue(np.allclose(Expected_Hak_myu,Hak_Myu,rtol=1e-3,atol=1e-4))

    def test_compute_Hkb(self):
        # Matrix simulated in MATLAB
        
        Expected_Hkb = np.array([[[4.49596505087895, 1.79758044633223, 1.11881609865121]]])

        P = self.HKI.as_intern_sys.compute_Pi(self.psig,self.HKI.Q)
        G = self.HKI.as_intern_sys.compute_Gi(self.psig,self.HKI.Q,P)

        Hkb_Psi = self.HKI.compute_Hkb(self.psig,G)

        self.HKI.switchMode()

        Hkb_Myu = self.HKI.compute_Hkb(self.psig,G)

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

        expected_C = np.array([[[4.4960,1.7976,1.1188]]])

        Hab,Habk,Hak,Hkb = self.HKI.TrueHoKalmanBase(self.psig)

        dLPV_sys = self.HKI.identify(Hab,Habk,Hak,Hkb)
        
        self.HKI.switchMode()

        Hab_M,Habk_M,Hak_M,Hkb_M = self.HKI.TrueHoKalmanBase(self.psig)

        dLPV_sys_Myu = self.HKI.identify(Hab_M,Habk_M,Hak_M,Hkb_M)

        self.assertTrue(np.allclose(expected_A,dLPV_sys.A,rtol=1e-3,atol=1e-4))
        self.assertTrue(np.allclose(expected_B,dLPV_sys.B,rtol=1e-3,atol=1e-4))
        self.assertTrue(np.allclose(expected_C,dLPV_sys.C,rtol=1e-3,atol=1e-4))
        self.assertTrue(np.allclose(self.D,dLPV_sys.D,rtol=1e-3,atol=1e-4))

    def test_seperate_Bsig(self):
        Bsig = np.zeros((2,3,2))
        
        Bsig[0,:,:] = [
            [1.1481 , 0.0625],
            [0.7220 , 0.0149],
            [-1.5496 , 0.0597]
        ]
        Bsig[1,:,:] = [
            [1.0000 ,-0.0279],
            [0 , 0.2029],
            [0, 0.1525]
        ]

        expected_Bi = np.zeros((2,3,1))
        
        expected_Bi[0,:,:] = [
            [1.1481],
            [0.722],
            [-1.5496]
        ]

        expected_Bi[1,:,:] = [
            [1],
            [0],
            [0]
        ]

        Gi = np.zeros((2,3,1))
        
        expected_Gi = np.zeros((2,3,1))

        expected_Gi[0,:,:] = [
            [0.0625],
            [0.0149],
            [0.0597]
        ]

        expected_Gi[1,:,:] = [
            [-0.0279],
            [0.2029],
            [0.1525]
        ]

        Bi,Gi = self.HKI.seperate_Bsig(Bsig)
 
        self.assertTrue(np.allclose(expected_Bi,Bi,rtol=1e-3,atol=1e-4))
        
        self.assertTrue(np.allclose(expected_Gi,Gi,rtol=1e-3,atol=1e-4))

    def test_compute_Tsig(self):
        
        expected_T_sig = np.zeros((2,1,1))
        expected_T_sig [0, :, :] = 1.9358
        expected_T_sig [1, :, :] = 2.9065

        self.HKI.switchMode()

        (Hab,Habk,Hak,Hkb) = self.HKI.TrueHoKalmanBase(self.psig)

        dLPV_sys = self.HKI.identify(Hab,Habk,Hak,Hkb)

        T_sig = self.HKI.compute_Tsig(dLPV_sys.A,dLPV_sys.C[0,:,:],dLPV_sys.B,self.psig)

        self.assertTrue(np.allclose(expected_T_sig,T_sig,rtol=1e-3,atol=1e-4))

    def test_compute_Tsig_det(self):
        expected_T_sig = np.zeros((2,1,1))
        expected_T_sig [0, : , :] = 124.2797
        expected_T_sig [1, : , :] = 124.2797

        HKI2 = HoKalmanIdentifier(self.A,self.B,self.C,self.D,self.K,self.Q_d,self.base)

        (Hab,Habk,Hak,Hkb) = HKI2.TrueHoKalmanBase(self.psig)

        dLPV_sys = HKI2.identify(Hab,Habk,Hak,Hkb)

        T_sig = HKI2.compute_Tsig(dLPV_sys.A,dLPV_sys.C[0,:,:],dLPV_sys.B,self.psig)
   
        self.assertTrue(np.allclose(expected_T_sig,T_sig,rtol=1e-3,atol=1e-4))
  
    def test_compute_K_Q_det(self):
        (Hab_d,Habk_d,Hak_d,Hkb_d) = self.HKI.TrueHoKalmanBase(self.psig)

        dLPV_sys = self.HKI.identify(Hab_d,Habk_d,Hak_d,Hkb_d)

        Q_old,K_old = self.HKI.compute_K_Q(dLPV_sys.A,dLPV_sys.B,dLPV_sys.C,self.psig)
        
        # No value in MATLAB to compare it with in the test, 
        # However, the values seem legitimate
    
    def test_compute_K_Q(self):
        expected_Q = np.zeros((2,1,1))

        expected_Q [0, : , :] = 1.5883

        expected_Q [1, : , :] = 2.5591

        expected_K = np.zeros((2,3,1))

        expected_K [0, : , :] = [
                [0.0120],
                [0.0029],
                [0.0101]
        ]
        
        expected_K [1, : , :] = [
                [-0.0094],
                [0.0451],
                [0.0530]
        ]
    

        self.HKI.switchMode()
    
        (Hab,Habk,Hak,Hkb) = self.HKI.TrueHoKalmanBase(self.psig)

        dLPV_sys = self.HKI.identify(Hab,Habk,Hak,Hkb)
        
        Q_old,K_old = self.HKI.compute_K_Q(dLPV_sys.A,dLPV_sys.B,dLPV_sys.C,self.psig)

        self.assertTrue(np.allclose(expected_K,K_old,rtol=1e-3,atol=1e-4))

        self.assertTrue(np.allclose(expected_Q,Q_old,rtol=1e-2,atol=1e-3))
