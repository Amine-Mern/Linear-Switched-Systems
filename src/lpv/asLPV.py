from .LPV import LPV
import numpy as np
import math

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
        self.ne = K.shape[1]
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
        
        b1 = max_abs_eigval < 1 - epsi
        b2 = np.array_equal(self.F, np.eye(self.ny))
        
        return b1 and b2
        
    def compute_vsp(self,v):
        """
        Computes the average outer product of global variable v.
        TESTED
        """
        Ntot = v.shape[1]
        ny = v.shape[0]
        v_esp = np.zeros((ny,ny))
        for i in range(Ntot):
            v_esp = v_esp + v[:,i] @ v[:,i].T
        v_esp /= Ntot
        return v_esp[0][0]

    def compute_Qi(self,v,p):
        """
        Compute the matrix Q_i = E[v(t) v(t)^T * mu_i(t)^2]
        TESTED
        """
        Q_true = np.zeros((self.ne, self.ne, self.np))
        Ntot = v.shape[1]
        for i in range(self.np):
            for t in range(Ntot):
                Q_true[:,:,i] = Q_true[:,:,i] + v[:,t] @ v[:,t].T * (p[i,t]**2)
            Q_true[:,:,i] = Q_true[:,:,i]/Ntot
        
        return Q_true
        
    def compute_Pi(self,psig,Q_true):
        """
        Computes the stationary covariance matrix P_i via iterative Lyapunov recursion.
        TESTED
        """
        P_true_old = np.zeros((self.nx, self.nx, self.np))
        P_true_new = np.zeros((self.nx,self.nx,self.np))
        for i in range(self.np):
            P_true_old[:,:,i] = np.zeros((self.nx,self.nx))
        
        max_ = np.ones((self.np,1))
        M_e = 1e-5 * np.ones(self.np)
        while np.any(max_ > M_e):
            for sig in range(self.np):
                P_true_new[:,:,sig] = np.zeros((self.nx,self.nx))
                for sig1 in range(self.np):
                    P_true_new[:,:,sig] += psig[sig,0] * (self.A[:,:,sig1] @ P_true_old[:,:,sig1]@ self.A[:,:,sig1].T + self.K[:,:,sig1] @ Q_true[:,:,sig1] @ self.K[:,:,sig1].T)
            for i in range(self.np):
                max_[i] = np.linalg.norm(P_true_new[:, :, i] - P_true_old[:, :, i], ord=2) / (np.linalg.norm(P_true_old[:, :, i], ord=2) + 1)
            
            P_true_old = P_true_new.copy()
            
        ## Putting it in the same MATLAB Shape :
        P_true_new_well_shaped = P_true_new.transpose(2,0,1)
        P_rounded = np.round(P_true_new_well_shaped,4)
        
        return P_rounded
    
    def computing_Gi(v,psig,p,Q_true,P_true_new):
        """
        Computes the matrix G_i used in the innovation form of the LPV system.
        UNTESTED
        """
        G_true = np.zeros((nx, ny, np_))
        for i in range(np):
            G_true[:,:,i] = (1/math.sqrt(psig[i,1])) @ (self.A[:,:,i] @ P_true_new[:,:,i] @ self.C.T + K[:,:,i] @ Q_true[:,:,i] @ F.T)
        return G_true
    
    def convertToDLPV():
        """
        Converts an asLPV to an dLPV
        """
    
    def stochMinimize():
        """
        Finds a minimal stochastic realization (in innovation form) of the current asLPV system.
        """
        
