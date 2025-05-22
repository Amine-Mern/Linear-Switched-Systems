import numpy as np
class LPV:
    """
    Base class representing a Linear Parmeter Varying systems.
    
    This concrete class defines the common structure for LPV system models,
    Especially useful for define specific LPV models such as (asLPV and dLPV..)
    
    (A CORRIGER)
    Parameters
    ----------------
    A : ndArray
         Constant Matrix in LPV system
    B : ndArray
         Constant Matrix in LPV system
    K : ndArray
         Constant Matrix in LPV system
    
    C : ndArray 
         Constnat Matrix in LPV system
    D : ndArray
         Constnat Matrix in LPV system
    F : ndArray
         Constnat Matrix in LPV system
    
    
    
    Methods
    ----------------
    Simulate_y()
    
    Simulate_Innovation()
    
    isEquivalent()
    
    """
    def __init__(self, A, B, C, D, K=None, F=None):
        """
        Constructor of the Linear Parameter Varying system
        
        Parameters:
            - A, B, C, D: system matrices (can be 3D arrays)
            - K, F: optional noise-related matrices
        """
        self.A = A
        self.B = B
        self.K = K
        self.C = C
        self.D = D
        self.F = F
        
        self.nx = A.shape[0]
        self.ny = C.shape[0]
        self.nu = D.shape[1]
        self.np = A.shape[2] if A.ndim == 3 else 1
        
    
    def simulate_y(self, u, v, p, Ntot):
        """
        Simulate the LPV system output
        
        Parameters:
            - u : ndarray
                  Input array
            - v : ndarray
                  Noise array, can be None if no noise
            - p : ndarray
                  Scheduling
            - Ntot : int
                     Total number of time steps
            
        Returns:
            - y : ndarray
                  Output with noise
            - ynf : ndarray
                    Output noise free
            - x : ndarray
                  State trajectory
        """
        nx, ny, np_ = self.nx, self.ny, self.np
        x = np.zeros((nx, Ntot))
        y = np.zeros((ny, Ntot))
        ynf = np.zeros((ny, Ntot))
        
        for k in range (Ntot-1):
            for i in range(np_):
                term_noise = 0
                if self.K is not None and v is not None:
                    term_noise = self.K[:,:,1] @ v[:,k]
                x[:, k+1] += (self.A[:, :, i] @ x[:, k] + self.B[:, :, i] @ u[:, k] + term_noise) * p[i, k] 
            noise_output = 0
            if self.F is not None and v is not None:
                noise_output = self.F @ v[:,k]
            y[:, k] = self.C @ x[:, k] + self.D @ u[:, k] + noise_output
            ynf[:, k] = self.C @ x[:, k]
        return y, ynf, x
    
    def simulate_Innovation(self):
        return None
    
    def isEquivalent(self):
        return None
    
        