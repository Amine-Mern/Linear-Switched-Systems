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
    
    
    def isFormInnovation():
        """
        Checks if the autonomous stochastic Linear Parameter Varying System is
        Innovation Form
        
        Returns : Boolean
        """
        
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
        