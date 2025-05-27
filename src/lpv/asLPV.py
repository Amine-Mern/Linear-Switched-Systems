from LPV import LPV
import numpy as np

class asLPV(LPV):
    """
    Class representing a autonomous stochastic Linear Parameter-Varying (asLPV) system.
    
    This class inherits from the base LPV class and is intended for modeling
    stochastic LPV systems (i.e., systems with stochastic noise inputs).
    
    Parameters
    ----------
    A : np.ndarray
        3D array representing the state transition matrices.
    K : np.ndarray
        3D array representing the process noise matrices.
    C : np.ndarray
        2D array representing the output matrix.
    F : np.ndarray
        2D array representing the measurement noise matrix.

    Attributes
    ----------
    nx : int
        Number of states.
    ny : int
        Number of outputs.
    nu : int
        Number of inputs.
    np : int
        Number of scheduling parameters.

    Methods
    -------
    
    isFormInnovation()
    
    convertToDLPV()
    
    StochMinimize()
    
    computingGi()
    
    compute_vesp()
    
    compute_Qi()
    
    compute_Pi()
    """
    
    def __init__(self, A, C, K, F):
        super().__init__(A, C, K=K, F=F, B=None, D=None,)
    
    
    def isFormInnovation(self,psig):
        """
        Checks if the autonomous stochastic Linear Parameter Varying System is
        Innovation Form
        
        Returns : Boolean
        """
        M = np.zeros((self.nx**2,self.nx**2))
        for i in range(1,self.np):
            A_KC = self.A[:,:,i]-self.K[:,:,i] @ self.C
            M += psig[i] * np.kron(A_KC,A_KC);
        epsi = 10**(-5)
                
        max_abs_eigval = max(abs(np.linalg.eigvals(M))) #Maximal absolut eigen value
        print(max_abs_eigval)
        print(M)
        
        b1 = max_abs_eigval < 1 - epsi
        b2 = np.array_equal(self.F, np.eye(self.ny))
        
        return b1 and b2
        
    def Computing_Gi():
        """
        Computes the matrix G_i used in the innovation form of the LPV system.
        """
    
    def Compute_Pi():
        """
        Computes the stationary covariance matrix P_i via iterative Lyapunov recursion.
        """
    
    def Compute_Qi():
        """
        Compute the matrix Q_i = E[v(t) v(t)^T * mu_i(t)^2]
        """
    
    def stochMinimize():
        """
        Finds a minimal stochastic realization (in innovation form) of the current asLPV system.
        """
        