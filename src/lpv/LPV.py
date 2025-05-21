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
    def __init__(self,A,B,K,C,D,F):
        """
        Constructor of the Linear Parameter Varying system
        """
        self.A = A
        self.B = B
        self.K = K
        self.C = C
        self.D = D
        self.F = F
    
    
    def simulate_y(self):
    
    def simulate_Innovation(self):
    
    def isEquivalent(self):
    
        