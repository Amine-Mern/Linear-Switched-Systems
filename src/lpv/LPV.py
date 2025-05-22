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
        
    
    def simulate_y(self):
        return None
    
    def simulate_Innovation(self):
        return None
    
    def isEquivalent(self):
        return None
    
        