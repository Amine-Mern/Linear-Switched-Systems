import unittest
import numpy as np
from src.lpv.asLPV import asLPV
from src.main.config import initialize_parameters

class asLPVTest(unittest.TestCase):
    def setUp(self):
        params  =  initialize_parameters()
        self.A, self.C, self.K, self.F  = params['A'], params['C'], params['K'], params['F']
        self.psig = params['psig']
        self.asLPV = asLPV(self.A,self.C,self.K,self.F)
        self.Ntot = params['Ntot']
    
    def test_isFormInnovation_True(self):
        self.assertTrue(self.asLPV.isFormInnovation(self.psig))
    
    def test_isFormInnovation_False_F_not_Identity(self):
        newF = np.array([[2]])
        asys2 = asLPV(self.A,self.C,self.K,newF)
        self.assertFalse(asys2.isFormInnovation(self.psig))
        
    def test_isFormInnovation_False_Unstable_eigValues(self):
        Unstable_A = np.zeros((2, 2, 2))
        Unstable_A[:, :, 0] = np.array([[1.2, 0.0],
                       [0.0, 1.1]])
        Unstable_A[:, :, 1] = np.array([[0.9, 0.4],
                       [0.3, 0.8]])

        Unst_K = np.zeros((2, 1, 2))
        Unst_K[:, :, 0] = np.array([[2.0],
                       [-1.0]])
        Unst_K[:, :, 1] = np.array([[1.5],
                       [1.5]])

        Unst_C = np.array([[1.0, 1.0]])
        Unst_F = np.array([[1.0]])
        psig = np.array([0.5, 0.5])

        asLPVsys = asLPV(Unstable_A,Unst_C,Unst_K,Unst_F)
        self.assertFalse(asLPVsys.isFormInnovation(psig))

    def test_Compute_vesp(self):
        #v_test : the v used in this test to check if the calculations are correct (length is shorter that normal) Ntot = 10 in this specific case, can't test 2000 length vector)
        v_test = np.array([[0.9794, -0.2656, -0.5484, -0.0963, -1.3807, -0.7284,1.8860 ,-2.9414,0.9800 ,-1.1918]])
        
        #This is the value of the expected v_esp (in the Matlab code this code originated from, the value approximated 1.8366)
        expected = 1.8366
        self.assertTrue(self.asLPV.compute_vsp(v_test) == expected)
    
    def test_compute_Qi(self):
        #v_test : the v used in this test to check if the calculations are correct (length is shorter that normal) Ntot = 10 in this specific case, can't test 2000 length vector))
        #This is used to avoid the randomness in the noise when we define it
        v_test = np.array([[0.9794, -0.2656, -0.5484, -0.0963, -1.3807, -0.7284,1.8860 ,-2.9414,0.9800 ,-1.1918]])
        
        #Randomness is used to defined p, so we define one for this test
        p_test = np.array([
    [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
    [0.0387, -0.1185, -0.4488, -1.2149, -0.1990, 0.6277, -1.1521, -1.2657, -0.3922, -1.3991]
])        
        Qi = self.asLPV.compute_Qi(v_test,p_test)
        
        #Values obtained from compiling the working Matlab code 
       
        expected1 = 1.8366
        expected2 = 2.1871
        
        #approx 1.8366 in MATLAB
        self.assertTrue(Qi[0][0][0] == expected1)
        
        #approx 2.187 in MATLAB
        self.assertTrue(Qi[0][0][1] == expected2)
        
    def test_compute_Pi(self):
         #v_test : the v used in this test to check if the calculations are correct (length is shorter that normal) Ntot = 10 in this specific case, can't test 2000 length vector))
        #This is used to avoid the randomness in the noise when we define it
        v_test = np.array([[0.9794, -0.2656, -0.5484, -0.0963, -1.3807, -0.7284,1.8860 ,-2.9414,0.9800 ,-1.1918]])
        #Randomness is used to defined p, so we define one for this test

        p_test = np.array([
[1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
[0.0387, -0.1185, -0.4488, -1.2149, -0.1990, 0.6277, -1.1521, -1.2657, -0.3922, -1.3991]
])
        Qi = self.asLPV.compute_Qi(v_test,p_test)

        psig_test = np.array([[1.0000],
        [0.7625]])
        
        Pi = self.asLPV.compute_Pi(psig_test,Qi)
        
        #Creating the expected Matrix showed in MATLAB :
        #Expected 3D Matrix
        #______________ Back Side______
        # Front side          | 0.8352 , 0.3517      
        # 1.0953 , 0.4612 | 0.3517 , 3.4438
        # 0.4612 , 4.5167 |--------------------
        
        Expected = np.zeros((2,2,2))
        Expected[:,:,0] = np.array([[1.0953,0.4612],[0.4612,4.5167]])
        Expected[:,:,1] = np.array([[0.8352,0.3517],[0.3517,3.4438]])
        
        Expected_rounded = np.round(Expected,4)

        self.assertTrue(np.allclose(Pi,Expected,rtol = 1e-4, atol = 1e-6))

    def test_compute_Gi(self):
        v_test = np.array([[0.9794, -0.2656, -0.5484, -0.0963, -1.3807, -0.7284,1.8860 ,-2.9414,0.9800 ,-1.1918]])
        #Randomness is used to defined p, so we define one for this test

        p_test = np.array([
[1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
[0.0387, -0.1185, -0.4488, -1.2149, -0.1990, 0.6277, -1.1521, -1.2657, -0.3922, -1.3991]
])
        psig_test = np.array([[1.0000],
        [0.7625]])
        
        ##Compute_Qi
        Qi = self.asLPV.compute_Qi(v_test,p_test)
        
        ##Compute_Pi
        Pi = self.asLPV.compute_Pi(psig_test,Qi)
        
        ##Compute_Gi
        Gi = self.asLPV.compute_Gi(v_test,psig_test,p_test,Qi,Pi)
        
        ## Expected :
        Expected = np.zeros((2, 1, 2)) 
        Expected[:, :, 0] = np.array([[0.6226],
                               [2.1018]])
        Expected[:, :, 1] = np.array([[0.1359],
                               [2.8169]])
        
        self.assertTrue(np.allclose(Gi,Expected,rtol = 1e-4, atol = 1e-6))
        
    def test_convertToDLPV(self):
        v_test = np.array([[0.9794, -0.2656, -0.5484, -0.0963, -1.3807, -0.7284,1.8860 ,-2.9414,0.9800 ,-1.1918]])
        
        #Randomness is used to defined p, so we define one for this test

        p_test = np.array([
[1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
[0.0387, -0.1185, -0.4488, -1.2149, -0.1990, 0.6277, -1.1521, -1.2657, -0.3922, -1.3991]
])
        psig_test = np.array([[1.0000],
        [0.7625]])
        
        d_system, T_sig = self.asLPV.convertToDLPV(v_test,p_test,psig_test)
        
        #self.assertTrue(False)
        
        
if __name__ == '__main__':
    unittest.main()