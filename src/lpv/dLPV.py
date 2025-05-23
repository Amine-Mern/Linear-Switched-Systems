from LPV import LPV

class dLPV(LPV):
    """
    Class representing a deterministic Linear Parameter-Varying (dLPV) system.
    
    This class inherits from the base LPV class and is intended for modeling
    deterministic LPV systems (i.e., systems without stochastic noise inputs).
    
    Parameters
    ----------
    A : np.ndarray
        3D array representing the state transition matrices
    B : np.ndarray
        3D array representing the input matrices.
    C : np.ndarray
        2D array representing the output matrix.
    D : np.ndarray
        2D array representing the feedthrough matrix.

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
    
    minimizeDLPV()
    
    reachRedLPV()
    
    obsRedLPV()
    
    isIsomorphic()
    
    Recursion()
    """
    
    def __init__(self, A, B, C, D):
        super().__init__(A, B, C, D, K=None, F=None)
        


