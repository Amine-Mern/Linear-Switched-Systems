import unittest
import numpy as np
from src.lpv.asLPV import asLPV

class asLPVTest(unittest.TestCase):
    
    def setUp(self):
        # These are the matrices are used in the main example in the MATLAB code this project is based on (this way we can test and verify the functions we define by comparing the values with the MATLAB
        # results)
        
        self.A = np.zeros((2,2,2))
        self.A[:, :, 0] = np.array([[0.4, 0.4], [0.2, 0.1]])
        self.A[:, :, 1] = np.array([[0.1, 0.1], [0.2, 0.3]])
        
        self.C = np.array([[1, 0]])
        
        self.K = np.zeros((2, 1, 2))
        self.K[:, :, 0] = np.array([[0], [1]])
        self.K[:, :, 1] = np.array([[0], [1]])
        
        self.F = np.array([[1]])
        
        self.asLPV = asLPV(self.A,self.C,self.K,self.F)        
        
        # These vectors/matrix are the ones computed in the main of MATLAB project with a set seed, we use these values same as MATLAB to compare them with our result values (so our results should
        # be equal to the ones in MATLAB)
        
        self.psig =  np.array([[1.0000],[0.7625]])
        self.p = np.array([[1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],[0.0387, -0.1185, -0.4488, -1.2149, -0.1990, 0.6277, -1.1521, -1.2657, -0.3922, -1.3991]]) 
        self.v = np.array([[0.9794, -0.2656, -0.5484, -0.0963, -1.3807, -0.7284,1.8860 ,-2.9414,0.9800 ,-1.1918]])
        
    def test_simulate_y(self):
        y,ynf,x = self.asLPV.simulate_y(self.v,self.p)
        
        # The simulated vectors in MATALAB
        expected_y = np.array([[0.9794, -0.2656, -0.1535, -0.0159, -1.4287, -0.7147, 1.3786, -3.5134, 0.8502, 0]])
        expected_ynf = np.array([[0, 0, 0.3949, 0.0804, -0.0480, 0.0137, -0.5074, -0.5720, -0.1298, 0]])
        expected_x = np.array([[0, 0, 0.3949, 0.0804, -0.0480, 0.0137, -0.5074, -0.5720, -0.1298, 0.2363], [0, 1.0174, -0.1686, -0.2529, 0.0841, -1.1102, -1.5012, 0.0973, 0.7848, 0.5660]])
        
        self.assertTrue(np.allclose(y,expected_y,rtol = 1e-4, atol = 1e-4))
        self.assertTrue(np.allclose(ynf,expected_ynf,rtol = 1e-4, atol = 1e-4))
        self.assertTrue(np.allclose(x,expected_x,rtol = 1e-4, atol = 1e-4))
    
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
        
        #This is the value of the expected v_esp (in the Matlab code this code originated from, the value approximated 1.8366)
        expected = 1.8366
        self.assertTrue(self.asLPV.compute_vsp(self.v) == expected)
    
    def test_compute_Qi(self):        
        Qi = self.asLPV.compute_Qi(self.v,self.p)
        
        #Values obtained from compiling the working Matlab code 
       
        expected1 = 1.8366
        expected2 = 2.1871
        
        #approx 1.8366 in MATLAB
        self.assertTrue(Qi[0][0][0] == expected1)
        
        #approx 2.187 in MATLAB
        self.assertTrue(Qi[0][0][1] == expected2)
        
    def test_compute_Pi(self):
        Qi = self.asLPV.compute_Qi(self.v,self.p)
        
        Pi = self.asLPV.compute_Pi(self.psig,Qi)
        
        #Creating the expected Matrix showed in MATLAB :
        #Expected 3D Matrix
        #______________ Back Side______
        # Front side          | 0.8352 , 0.3517      
        # 1.0953 , 0.4612 | 0.3517 , 3.4438
        # 0.4612 , 4.5167 |--------------------
        
        Expected = np.zeros((2,2,2))
        Expected[:,:,0] = np.array([[1.0953,0.4612],[0.4612,4.5167]])
        Expected[:,:,1] = np.array([[0.8352,0.3517],[0.3517,3.4438]])
        
        self.assertTrue(np.allclose(Pi,Expected,rtol = 1e-4, atol = 1e-6))

    def test_compute_Gi(self):
        ##Compute_Qi
        Qi = self.asLPV.compute_Qi(self.v,self.p)
        
        ##Compute_Pi
        Pi = self.asLPV.compute_Pi(self.psig,Qi)
        
        ##Compute_Gi
        Gi = self.asLPV.compute_Gi(self.psig,self.p,Qi,Pi)
        
        ## Expected :
        Expected = np.zeros((2, 1, 2)) 
        Expected[:, :, 0] = np.array([[0.6226],
                               [2.1018]])
        Expected[:, :, 1] = np.array([[0.1359],
                               [2.8169]])
        
        self.assertTrue(np.allclose(Gi,Expected,rtol = 1e-4, atol = 1e-6))
        
    def test_convertToDLPV(self):
        # We can construct all the expected matrix based on the correct MATLAB values
        expected_An = np.zeros((2,2,2))
        
        expected_An[:,:,0] = np.array([[0.4,0.4],[0.2,0.1]])
        expected_An[:,:,1] = np.array([[0.0873,0.0873],[0.1746,0.2619]])
        
        expected_Tsig = np.zeros((1,1,2))
        expected_Tsig[0,0,0] = 2.931
        expected_Tsig[0,0,1] = 3.9636
        
        expected_Fmin = np.eye(1)
        
        expected_C = np.zeros((1,2))
        expected_C[0,:] = np.array([1,0])
        
        expected_G_true = np.zeros((2,1,2))
        expected_G_true[:,0,0] = np.array([0.6226,2.1018])
        expected_G_true[:,0,1] = np.array([0.1359,2.8168])
        
        d_system, T_sig = self.asLPV.convertToDLPV(self.v,self.p,self.psig)
        
        An = d_system.A
        G_true = d_system.B
        C = d_system.C
        D = d_system.D
        
        self.assertTrue(np.allclose(An, expected_An, rtol=1e-03, atol=1e-6))
        self.assertTrue(np.allclose(G_true,expected_G_true, rtol=1e-06, atol=1e-8))
        self.assertTrue(np.allclose(C, expected_C, rtol=1e-06, atol=1e-8))
        self.assertTrue(np.allclose(D, expected_Fmin, rtol=1e-06, atol=1e-8))
        
        self.assertTrue(np.allclose(T_sig, expected_Tsig,rtol = 1e-03, atol = 1e-6))
        
    def test_stochminimize(self):
        #Execute the function we are testing
        as_min_system, Qmin = self.asLPV.stochMinimize(self.v,self.p,self.psig)
        
        A = as_min_system.A
        C = as_min_system.C
        K = as_min_system.K
        F = as_min_system.F
        
        exp_A = np.zeros((2, 2, 2))
        
        # By construction Amin can have negative values in the matrix caused by SVD
        # All singular values are distinct, so every valid SVD differs only by sign of each pair of singular vectors
        
        #Expected MATLAB A
        exp_A[:, :, 0] = np.array([[0.4645, -0.3579],[-0.1579,  0.0355]])
        exp_A[:, :, 1] = np.array([[0.1369, -0.1189],[-0.2189,  0.2631]])
        
        #We put Amin and A in absolute values since we want to check the values without the +/- sign
        exp_A_abs = np.abs(exp_A)
        A_abs = np.abs(A)
        #This way there will be no difference (up to a sign) between the valid SVDs
        #We use the exact same logic for K_min and C_min (*)

        #Expected values in MATLAB
        exp_C = np.zeros((1, 2))
        exp_C[0, :] = np.array([-0.9934, -0.1148])

        #(*)
        exp_C_abs = np.abs(exp_C)
        C_abs = np.abs(C)

        #Expected values in MATLAB
        exp_K = np.zeros((2, 1, 2))
        exp_K[:, 0, 0] = np.array([-0.1148, 0.9934])
        exp_K[:, 0, 1] = np.array([-0.1148, 0.9934])

        #(*)
        exp_K_abs = np.abs(exp_K)
        K_abs = np.abs(K)
        
        exp_F = np.eye(1)

        exp_Qmin = np.zeros((1, 1, 2))
        exp_Qmin[0, 0, 0] = 1.8366
        exp_Qmin[0, 0, 1] = 2.1872
        
        self.assertTrue(np.allclose(A_abs,exp_A_abs,rtol=1e-03, atol=1e-6))
        self.assertTrue(np.allclose(C_abs,exp_C_abs,rtol=1e-03, atol=1e-6))
        self.assertTrue(np.allclose(K_abs,exp_K_abs,rtol=1e-03, atol=1e-6))
        self.assertTrue(np.allclose(F,exp_F,rtol=1e-03, atol=1e-6))
         
        self.assertTrue(np.allclose(Qmin,exp_Qmin,rtol=1e-03, atol=1e-6))
        
if __name__ == '__main__':
    unittest.main()