import numpy as np
class LPV:
    """
    Base class representing a Linear Parmeter Varying systems.
    
    This concrete class defines the common structure for LPV system models,
    Especially useful for define specific LPV models such as (asLPV and dLPV..)
    
    Parameters
    ----------------
    A : np.ndarray
        3D array representing the state transition matrices.
    B : np.ndarray
        3D array representing the input matrices.
    C : np.ndarray
        2D array representing the output matrix.
    D : np.ndarray
        2D array representing the feedthrough matrix.
    K : np.ndarray, optional
        3D array representing the process noise matrices. Default is None.
    F : np.ndarray, optional
        2D array representing the measurement noise matrix. Default is None.
    
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
    simulate_y(u, v, p, Ntot)
        Simulates the LPV system output over a time horizon.
    
    simulate_Innovation(y,p,Ntot)
        Simulates the innovation error of an LPV system in innovation form

    isEquivalent(other, x0, x0_other, tolerance=1e-5)
        Checks whether this LPV system is equivalent to another LPV system
        by comparing their Markov parameters.
    
    """
    def __init__(self, A, C, B = None, D = None, K = None, F=None):
        """
        Constructor of the Linear Parameter Varying system
        
        Parameters:
            - A, B, C: system matrices (can be 3D arrays)
            - D, K, F: optional noise-related matrices
        """
        self.A = A
        self.B = B
        self.K = K
        self.C = C
        self.D = D
        self.F = F
        
        self.nx = A.shape[0]
        self.ny = C.shape[0]
        
        if (D != None) :
            self.nu = D.shape[1]
        else :
            self.nu = 0
        
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
                term_noise = self.K[:,:,1] @ v[:,k]
                x[:, k+1] += (self.A[:, :, i] @ x[:, k] + self.B[:, :, i] @ u[:, k] + term_noise) * p[i, k] 
            noise_output = self.F @ v[:,k]
            y[:, k] = self.C @ x[:, k] + self.D @ u[:, k] + noise_output
            ynf[:, k] = self.C @ x[:, k]
        return y, ynf, x
    
    def simulate_Innovation(self,Ntot,y,p):
        """
        If used on :
        asLPV : Simulates the innovation error of an LPV system in innovation form
        dLPV : Simulates the contribution of the input to the output 
        
        Parameters :
        - p : ndarray
                  Scheduling vector
        
        - y : ndarray
                  Output with noise
        
        Returns :
        - Res : Innovation error noise (if used on as-LPV (F @ v(t)))
                 : Static contribution of the input to(the output (if used on dLPV (D @ u(t)))
        
        """
        np_ = self.np
        x1 = np.zeros((self.nx,Ntot+1))
        Res = np.zeros((self.ny,Ntot))
        for k in range(Ntot):
            Res[:,k] = y[:,k] - self.C @ x1[:,k]
            for i in range(np_):
                x1[:,k+1] += (self.A[:,:,i] @ x1[:,k] + self.K[:,:,i] @ Res[:,k]) * p[i,k];
        
        Res[:,-1] = y[:,-1] - self.C @ x1[:,-1]
        return Res
    
    def isEquivalentTo(self, other, x0, x0_other, tolerance=None):
        """
        Check if this LPV is equivalent to another LPV system by comparing their
        Markov parameters up to a certain order.
        
        Parameters:
            other : LPV
                    Another LPV instance to compare against.
            x0    : ndarray
                    Initial state vector of this LPV system
            x0_other : ndarray
                       Initial state vector of the other LPV system.
            tolerance : float, optional
                        Numerical tolerance for equivalence check, default is 1e-5.
                        
        Returns:
            bool: True if the two LPV systems are equivalent within the tolerance, False otherwise.
        """
        if tolerance is None:
            tolerance = 1e-5
            
        n1, n2 = self.nx, other.nx
        D1, D2 = self.np, other.np
        
        if D1 != D2:
            raise ValueError("The two systems should have the same number of dimensions")
        
        D= D1
        ny = self.ny
        N = 2 * max(n1, n2) - 1

        x0_col = x0.reshape(-1, 1)
        x0_other_col = x0_other.reshape(-1, 1)

        R = np.vstack([
            np.hstack([x0_col] + [self.B[:, :, i] for i in range(D)]),
            np.hstack([x0_other_col] + [other.B[:, :, i] for i in range(D)])
        ])

        compC = np.hstack([self.C, -other.C])

        for step in range(N + 1):
            diff_markov = compC @ R # C1*(B1 or A1^k B1)-C2*(B2 or A2^k B2)
            if np.linalg.norm(diff_markov, 2) > tolerance:
                print("Not equivalent - Difference in Markov parameters: ")
                print(diff_markov)
                return False
            
            R_new = []
            for i in range(D):
                A1_block = self.A[:, :, i]
                A2_block = other.A[:, :, i]
                block = np.block([
                    [A1_block, np.zeros((n1, n2))],
                    [np.zeros((n2, n1)), A2_block]
                ])
                R_new.append(block @ R)
            R = np.hstack(R_new)
        return True


        

