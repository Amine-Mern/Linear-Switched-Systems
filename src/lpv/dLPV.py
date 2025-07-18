from .LPV import LPV
import numpy as np
from scipy.linalg import orth

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
    
    simulate_y()
    
    minimizeDLPV()
    
    reach_reduction()
    
    obsRedLPV()
    
    isIsomorphic()
    
    Recursion()

    convert_to_asLPV()
    """
    
    def __init__(self, A, C, B, D):
        super().__init__(A, C, B, D, K=None, F=None)
        
    def simulate_y(self, u, p, Ntot):
        """
        Simulate the dLPV system output
        
        Parameters:
            - u : ndarray
                  Input array
            - p : ndarray
                  Scheduling
            - Ntot : int
                     Total number of time steps
            
        Returns:
            - y : ndarray
                  Output with noise
            - yif : ndarray
                    Output input free
            - x : ndarray
                  State trajectory
        TESTED
        """
        nx, ny, np_ = self.nx, self.ny, self.np
        x = np.zeros((nx, Ntot))
        y = np.zeros((ny, Ntot))
        yif = np.zeros((ny, Ntot))
        
        for k in range (Ntot-1):
            for i in range(np_):
                x[:, k+1] += (self.A[i, :, :] @ x[:, k] + self.B[i, :, :] @ u[:, k]) * p[i, k] 
            y[:, k] = self.C @ x[:, k] + self.D @ u[:, k]
            yif[:, k] = self.C @ x[:, k]
        return y, yif, x
    
    def reach_reduction(self, x0):
        """
        Perform reachability reduction of the dLPV system
        
        Parameters:
            x0 : initial state vector
            
        Returns:
            Reach_mat : ndarray
                        Reachability matrix.
            reduced_ord : int
                          Reduced order.
            Ar : ndarray
                 Reduced A matrix
            Br : ndarray
                 Reduced B matrix
            Cr : ndarray
                 Reduced C matrix
            x0r : ndarray
                  Reduced initial state vector
        TESTED
        """
        Anum = np.vstack([self.A[i, :, :] for i in range(self.np)])
        
        Bnum = np.hstack([self.B[i, :, :] for i in range(self.np)])
        
        Cnum = np.vstack([self.C for _ in range(self.np)])
        x0 = x0.reshape(-1, 1)
        B_hat = np.hstack([x0, Bnum])
        
        V_f = orth( B_hat)
        
        if V_f.shape[1] == 0:
            print(
                "ERROR : Reachability matrix is empty. "
                "This usually means that the input matrix B (or gain K in innovation form) is zero, "
                "and the initial state x0 does not excite the system. "
                "The system is not reachable under the given conditions."
            )
            exit()
        
        V_0 = V_f.copy()
        
        quit = False
        while not quit :
            r = np.linalg.matrix_rank(V_f)
            V_prime = V_0.copy()
            
            for j in range(self.np):
                Aj = Anum[j * self.nx:(j + 1) * self.nx, :]
                V_prime = np.hstack([V_prime, Aj @ V_f])
            V_f = orth(V_prime)
            quit = (r == np.linalg.matrix_rank(V_f))
        
        Reach_mat = V_f
        reduced_order = Reach_mat.shape[1]

        Ar = np.zeros((self.np, reduced_order, reduced_order))
        Br = np.zeros((self.np, reduced_order, self.nu))
        
        for i in range(self.np):
            Ar[i, :, :] = Reach_mat.T @ self.A[i, :, :] @ Reach_mat
            Br[i, :, :] = Reach_mat.T @ self.B[i, :, :]

        Cr = self.C @ Reach_mat
        x0r = Reach_mat.T @ x0
        
        return Reach_mat, reduced_order, Ar, Br, Cr, x0r

    
    def obs_reduction(self, x0):
        """
        Perform observability reduction of the dLPV system.

        Parameters:
            x0 : ndarray
                 Initial state vector 

        Returns:
            Wf : ndarray
                 Observability matrix 
            reduced_order : int
                            Dimension of the reduced observable system
            Ao : ndarray
                 Reduced A matrix 
            Bo : ndarray
                 Reduced B matrix 
            Co : ndarray
                 Reduced C matrix 
            x0o : ndarray
                  Reduced initial state
        TESTED
        """
        Cnum = np.vstack([self.C for _ in range(self.np)])
        x0 = x0.reshape(-1,1)
        Wf = orth(Cnum.T)
        r = np.linalg.matrix_rank(Wf)

        quit = False
        while not quit:
            Wprime = Cnum.T
            for i in range(self.np):
                Wprime = np.hstack((Wprime, self.A[i, :, :].T @ Wf))
            new_Wf = orth(Wprime)
            quit = new_Wf.shape[1] == r
            r = new_Wf.shape[1]
            Wf = new_Wf

        reduced_order = r

        Ar = np.zeros((self.np, reduced_order, reduced_order))
        Br = np.zeros((self.np, reduced_order, self.B.shape[2]))
        for i in range(self.np):
            Ar[i, :, :] = Wf.T @ self.A[i, :, :] @ Wf
            Br[i, :, :] = Wf.T @ self.B[i, :, :]

        Cr = self.C @ Wf
        x0r = Wf.T @ x0

        Obs_mat = Wf
        
        return Obs_mat, reduced_order, Ar, Br, Cr, x0r
    
    def minimize(self, x0):
        """
        Perform minimal realization reduction of the dLPV system.

        Parameters:
            x0 : ndarray
                 Initial state vector.

        Returns:
            minimal_sys : dLPV
                          Reduced minimal dLPV system.
            x0m : ndarray
                  Reduced initial state vector.
        TESTED
        """
        x0 = x0.reshape(-1,1)
        
        Reach_mat, red_ord_r, Ar, Br, Cr, x0r = self.reach_reduction(x0)
        
        temp_sys = dLPV(Ar, Cr, Br, np.zeros((self.ny, self.nu)))
        Obs_mat, red_ord_m, Ao, Bo, Co, x0m = temp_sys.obs_reduction(x0r)
        
        minimal_sys = dLPV(Ao, Co, Bo, np.zeros((self.ny, self.nu)))
        return minimal_sys, x0m
    
    
    
    def isIsomorphic(self, other, x0, x0_other):
        """
        Check whether two dLPV systems are isomorphic (equivalent up to a change of state basis).

        Parameters:
            other : dLPV
                Another LPV system to compare with.
            x0 : ndarray
                 Initial state vector for the dLPV.
            x0_other : ndarray
                 Initial state vector for the other dLPV.    

        Returns:
            bool : True if systems are isomorphic, False otherwise.
        TESTED
        """
        tol = 1e-10
        if not isinstance(other, dLPV):
            raise ValueError("Other system must be an instance of dLPV.")
        if self.nx != other.nx or self.np != other.np:
            return False
        
        Anum3 = np.zeros((self.np, 2 * self.nx, 2 * self.nx))
        Bnum3 = np.zeros((self.np, 2 * self.nx, self.nu))
        
        
        for i in range(self.np):
            Anum3[i, :, :] = np.block([
                [self.A[i, :, :], np.zeros((self.nx, self.nx))],
                [np.zeros((self.nx, self.nx)), other.A[i, :, :]]
            ])
            Bnum3[i, :, :] = np.vstack([
                self.B[i, :, :],
                other.B[i, :, :]
            ])
        
        Cnum3 = np.hstack([self.C, other.C])

        Dnum3 = np.hstack([self.D, other.D])

        x03 = np.vstack([x0.reshape(-1, 1), x0_other.reshape(-1, 1)])
        
        merged_dLPV = dLPV(Anum3, Cnum3, Bnum3,self.D)
        obs_mat, obs_rank, Ao, Bo, Co, x0o = merged_dLPV.obs_reduction(x03)

        S = obs_mat.T
        T1 = S[:, :self.nx]
        T2 = S[:, self.nx:]

        try:
            T = np.linalg.inv(T2) @ T1
        except np.linalg.LinAlgError:
            return False  

        for i in range(self.np):
            Ai1 = self.A[i, :, :]
            Ai2 = other.A[i, :, :]
            if np.linalg.norm(T @ Ai1 - Ai2 @ T) > tol:
                return False

            Bi1 = self.B[i, :, :]
            Bi2 = other.B[i, :, :]
            if np.linalg.norm(T @ Bi1 - Bi2) > tol:
                return False

        if np.linalg.norm(self.C - other.C @ T) > tol:
            return False

        if np.linalg.norm(T @ x0.reshape(-1, 1) - x0_other.reshape(-1, 1)) > tol:
            return False

        return True
        
        
        
    def Recursion(self, T_sig, psig):
            """
            Performs the LPV recursion to estimate the matrices P, Q, and K.

            Parameters:
                T_sig : ndarray [ny, ny, np]
                    Measurement noise covariance matrices for each LPV mode.
                psig : ndarray [np, 1]
                    Probability weights for each mode.

            Returns:
                Pold : ndarray [nx, nx, np]
                Qold : ndarray [ny, ny, np]
                Kold : ndarray [nx, ny, np]
            TESTED
            """
            A = self.A
            B = self.B
            C = self.C

            nx = self.nx
            ny = self.ny
            np_ = self.np

            Pold = np.zeros((np_, nx, nx))
            Qold = np.zeros((np_, ny, ny))
            Kold = np.zeros((np_, nx, ny))
            max_err = np.ones(np_)
            while np.any(max_err > 1e-8):

                for sig in range(np_):
                    Qold[sig, :, :] = psig[sig, 0] * T_sig[sig, :, :] - C @ Pold[sig, :, :] @ C.T
                    
                    invQ = np.linalg.inv(Qold[sig, :, :])
                    sqrt_psig = np.sqrt(psig[sig, 0])
                    inv_sqrt_psig = 1.0 / sqrt_psig
                    
                    Kold[sig, :, :] = (sqrt_psig * B[sig, :, :] - inv_sqrt_psig * A[sig, :, :] @ Pold[sig, :, :] @ C.T) @ invQ
                    Pnew = np.zeros_like(Pold)
                for sig in range(np_):
                    for sig1 in range(np_):
                        term1 = (1 / psig[sig1, 0]) * A[sig1, :, :] @ Pold[sig1, :, :] @ A[sig1, :, :].T
                        term2 = Kold[sig1, :, :] @ Qold[sig1, :, :] @ Kold[sig1, :, :].T
                        Pnew[sig, :, :] += psig[sig, 0] * (term1 + term2)

                for i in range(np_):
                    num = np.linalg.norm(Pnew[i, :, :] - Pold[i, :, :])
                    den = np.linalg.norm(Pold[i, :, :]) + 0.1
                    max_err[i] = num / den

                Pold = np.copy(Pnew)

            return Pold, Qold, Kold

    def convert_to_asLPV(self, T_sig, psig):
        from .asLPV import asLPV 
        """
        Converts the current dLPV system to an asLPV system using stochastic recursion.

        Parameters:
            T_sig : ndarray [ny, ny, np]
                Measurement noise covariance matrices for each mode.
            psig : ndarray [np, 1]
                Probability weights for each mode.

        Returns:
            as_system : asLPV
                Instance of asLPV class with updated A, K, C, F.
            Qmin : ndarray [ny, ny, np]
                Output noise covariance matrices computed from recursion.
        TESTED
        """
        Pmin, Qmin, Kmin = self.Recursion(T_sig, psig)
        Amin = np.zeros_like(self.A)
        for i in range(self.np):
            Amin[i, :, :] = (1.0 / np.sqrt(psig[i, 0])) * self.A[i, :, :]

        F = np.eye(self.ny)
        
        # We round to 4 decimal point same as in the original Matlab code
        Amin = np.round(Amin,4)
        Kmin = np.round(Kmin,4)
        Qmin = np.round(Qmin,4)
        self.C = np.round(self.C,4)
        
        as_system = asLPV(Amin, self.C, Kmin, F)

        return as_system, Qmin
