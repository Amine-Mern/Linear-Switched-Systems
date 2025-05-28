from LPV import LPV
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
        UNTESTED
        """
        Ntot = v.shape[1]
        ny = v.shape[0]
        v_esp = np.zeros((ny,ny))
        for i in range(Ntot):
            v_esp = v_esp + v[:,i] @ v[:,i].T
        v_esp /= Ntot
        return v_esp

    def Compute_Qi(self,v,p):
        """
        Compute the matrix Q_i = E[v(t) v(t)^T * mu_i(t)^2]
        UNTESTED
        """
        Q_true = np.zeros((self.ne, self.ne, self.np))
        Ntot = v.shape[1]
        for i in range(self.np):
            for t in range(self.Ntot):
                Q_true[:,:,i] = Q_true[:,:,i] + v[:,i].reshape(-1,1) @ v[:,i].reshape(1,-1) * (p[i,t]**2)
            Q_true[:,:,i] = Q_true[:,:,i]/Ntot
        return Q_true
        
    def Compute_Pi(self):
        """
        Computes the stationary covariance matrix P_i via iterative Lyapunov recursion.
        UNFINISHED
        """
        P_true_old = np.zeros((self.nx, self.nx, self.np))
        P_true_new = np.zeros((self.nx,self.nx,self.np))
        for i in range(self.np):
            P_true_old[:,:,i] = zeros(self.nx,self.nx)
        
        max_ = np.ones(np,1)
        while max_ > 10**(-5) *  np.ones(self.np):
            for sig in range(np):
                P_true_new[:,:,sig] = zeros(self.nx,self.nx)
                for sig1 in range(np):
                    P_true_new[:,:,sig] = P_true_new[:,:,sig] + psig[sig,1] @ (A[:,:,sig1] @ P_true_old[:,:,sig1]@ A[:,:,sig1].reshape(-1,1) + k[:,:,sig1] @ Q_true[:,:,sig1] @ K[:,:,sig1].reshape(-1,1))
            for i in range(self.np):
                max_[i] = norm(P_true_new[:,:,i] - P_true_old[:,:,i])/norm(P_true_old[:,:,sig1] @ A[:,:,sig1].T + K[:,:,sig1] @ Q_true[:,:,sig1] @ K[:,:,sig1].T)
            return P_true_new
    
    def Computing_Gi(v,psig,p,Q_true,P_true_new):
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
        