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
    simulate_y()
    
    isFormInnovation()
    
    convertToDLPV()
    
    StochMinimize()
    
    computingGi()
    
    compute_vesp()
    
    compute_Qi()
    
    compute_Pi()
    """
    
    def __init__(self, A, C, K, F):
        self.ne = K.shape[2]
        super().__init__(A, C, K=K, F=F, B=None, D=None,)
    
    def simulate_y(self, v, p):
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
        TESTED
        """
        nx, ny, np_ ,Ntot = self.nx, self.ny, self.np,v.shape[1]
        x = np.zeros((nx, Ntot))
        y = np.zeros((ny, Ntot))
        ynf = np.zeros((ny, Ntot))
        
        for k in range (Ntot-1):
            for i in range(np_):
                term_noise = self.K[i,:,:] @ v[:,k]
                x[:, k+1] += (self.A[i, :, :] @ x[:, k] + term_noise) * p[i, k] 
            noise_output = self.F @ v[:,k]
            y[:, k] = self.C @ x[:, k] + noise_output
            ynf[:, k] = self.C @ x[:, k]
        
        round_y, round_ynf, round_x = np.round(y,4),np.round(ynf,4),np.round(x,4)
        return round_y,round_ynf,round_x
    
    
    def isStablyInvertable(self,psig):
        """
        Checks if the autonomous stochastic Linear Parameter Varying System is
        stably invertable

        Parameters:
            psig : ndarray [np,1]
                Probability weights for each mode
        
        Returns : Boolean
        TESTED
        """
        M = np.zeros((self.nx**2,self.nx**2))
        for i in range(1,self.np):
            A_KC = self.A[i,:,:]-self.K[i,:,:] @ self.C
            M += psig[i] * np.kron(A_KC,A_KC);
        epsi = 10**(-5)
                
        max_abs_eigval = max(abs(np.linalg.eigvals(M))) #Maximal absolut eigen value
        
        b1 = max_abs_eigval < 1 - epsi
        b2 = np.array_equal(self.F, np.eye(self.ny))
        
        return b1 and b2
        
    def compute_vsp(self,v):
        """
        Computes the average outer product of global variable v.
        
        Parameters :
            v : ndarray
                Noise array

        Returns :
            vesp : ndarray[ny,Ntot]

        TESTED
        """
        Ntot = v.shape[1]
        ny = v.shape[0]
        v_esp = np.zeros((ny,ny))
        for i in range(Ntot):
            v_esp = v_esp + v[:,i] @ v[:,i].T
        v_esp /= Ntot
        
        ## We round the number to four decimal places (as used the original MATLAB code)
        return round(v_esp[0][0],4)

    def compute_Qi(self,v,p):
        """
        Compute the matrix Q_i = E[v(t) v(t)^T * mu_i(t)^2]
        
        Parameters :
            v : ndarray
                  Noise array, can be None if no noise
            p : ndarray
                  Scheduling
        
        Returns :
            Qi : ndArray

        TESTED
        """
        Q_true = np.zeros((self.np, self.ne, self.ne))
        Ntot = v.shape[1]
        for i in range(self.np):
            for t in range(Ntot):
                Q_true[i,:,:] = Q_true[i,:,:] + v[:,t] @ v[:,t].T * (p[i,t]**2)
            Q_true[i,:,:] = Q_true[i,:,:]/Ntot
        
        return np.round(Q_true,4)
        
    def compute_Pi(self,psig,Q_true):
        """
        Computes the stationary covariance matrix P_i via iterative Lyapunov recursion.
        
        Parameters:
            psig : ndarray [np,1]
                Probability weights for each mode

            Q_true : ndarray
                the array compute thanks to compute_Qi
        
        Returns :
            Pi : ndarray

        TESTED
        """
        P_true_old = np.zeros((self.np, self.nx, self.nx))
        P_true_new = np.zeros((self.np,self.nx,self.nx))
        for i in range(self.np):
            P_true_old[i,:,:] = np.zeros((self.nx,self.nx))
        
        max_ = np.ones((self.np,1))
        M_e = 1e-5 * np.ones(self.np)
        while np.any(max_ > M_e):
            for sig in range(self.np):
                P_true_new[sig,:,:] = np.zeros((self.nx,self.nx))
                for sig1 in range(self.np):
                    P_true_new[sig,:,:] += psig[sig,0] * (self.A[sig1,:,:] @ P_true_old[sig1,:,:]@ self.A[sig1,:,:].T + self.K[sig1,:,:] @ Q_true[sig1,:,:] @ self.K[sig1,:,:].T)
            for i in range(self.np):
                max_[i] = np.linalg.norm(P_true_new[i, :, :] - P_true_old[i, :, :], ord=2) / (np.linalg.norm(P_true_old[i, :, :], ord=2) + 1)
            
            P_true_old = P_true_new.copy() 

        P_true_new_rounded = np.round(P_true_new,4)
        
        return P_true_new_rounded
    
    def compute_Gi(self,psig, Q_true,P_true_new):
        """
        Computes the matrix G_i used in the innovation form of the LPV system.
        Parameters:
            psig : ndarray [np,1]
                Probability weights for each mode

            Q_true : ndarray
                the array computed thanks to compute_Qi

            P_true_new : ndarray
                the array computed thanks to compute_Pi 

        TESTED
        """
        G_true = np.zeros((self.np, self.nx, self.ny))

        for i in range(self.np):
            G_true[i,:,:] = (1/math.sqrt(psig[i,0])) * (self.A[i,:,:] @ P_true_new[i,:,:] @ self.C.T + self.K[i,:,:] @ Q_true[i,:,:] @ self.F.T)
        
        G_true_rounded = np.round(G_true,4)
        return G_true_rounded
    
    def convertToDLPV(self,v,p,psig):
        from .dLPV import dLPV
        """
        Converts an asLPV to an dLPV
        We calculate T_sig here to use P_True, Q_True, without needing to use them later.

        Parameters :
            v : ndarray
                Noise array
                
            p : ndarray
                Scheduling array

            psig : ndarray [np,1]
                Probability weights for each mode
        
        Returns :
            d_system : DLPV
                The associated DLPV to the current asLPV

            Tsig : ndarray

        TESTED
        """
        Q_true = self.compute_Qi(v,p)
        P_true = self.compute_Pi(psig,Q_true)
        G_true = self.compute_Gi(psig,Q_true,P_true)
        
        # Matrix we wish to calculate
        T_sig = np.zeros((self.np,self.ny,self.ny))
        An = np.zeros((self.np,self.nx,self.nx))
        
        for i in range(self.np):
            T_sig[i,:,:] = (1/psig[i,0]) * (self.C @ P_true[i,:,:] @ self.C.T + self.F @ Q_true[i,:,:] @ self.F.T)
            An[i,:,:] = (math.sqrt(psig[i,0]) * self.A[i,:,:])
        
        # We create a DLPV System with arguments such as
        # A : An
        # B : G_True
        # C : Stays as C
        # D : Fmin a square eye matrix of dimension ny
        
        Fmin = np.eye(self.ny)
        
        #We round the two matrices that are not rounded to 4 decimal point
        T_sig = np.round(T_sig,4)
        An = np.round(An,4)
        
        d_system = dLPV(An,self.C,G_true,Fmin)
        
        return d_system , T_sig
    
    def stochMinimize(self,v,p,psig):
        """
        Finds a minimal stochastic realization (in innovation form) of the current asLPV system.
        Qmin is later used in main
        
        Paramaters :
            v : ndarray
                Noise array
                
            p : ndarray
                Scheduling array

            psig : ndarray [np,1]
                Probability weights for each mode

        Returns :
            as_min_system : asLPV
                The minimized asLPV system
            Qmin : ndarray

        TESTED
        """
        x0 = np.zeros((self.nx,1))
                
        d_system, T_sig = self.convertToDLPV(v,p,psig)
        
        min_d_system, x0m = d_system.minimize(x0)
        
        as_min_system, Qmin = min_d_system.convert_to_asLPV(T_sig,psig)
        
        return as_min_system, Qmin
    
