from LPV import LPV
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
    
    minimizeDLPV()
    
    reach_reduction()
    
    obsRedLPV()
    
    isIsomorphic()
    
    Recursion()
    """
    
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
        """
        Anum = np.vstack([self.A[:, :, i] for i in range(self.np)])
        print("A = ",Anum)
        Bnum = np.hstack([self.B[:, :, i] for i in range(self.np)])
        print("B = ",Bnum)
        Cnum = np.vstack([self.C for _ in range(self.np)])
        print("C = ",Cnum)
        x0 = x0.reshape(-1, 1)
        B_hat = np.hstack([x0, Bnum])
        print("B_hat", B_hat)
        print("x_0", x0)
        
        V_f = orth( B_hat,1e-6)
        print("V_f = ", V_f)
        V_0 = V_f.copy()
        print("V_0 = ", V_0)
        
        quit = False
        while not quit :
            r = np.linalg.matrix_rank(V_f)
            print("r", r)
            V_prime = V_0.copy()
            print("V_prime", V_prime)
            for j in range(self.np):
                Aj = Anum[j * self.nx:(j + 1) * self.nx, :]
                V_prime = np.hstack([V_prime, Aj @ V_f])
                print("V_prim ", V_prime, j)
            print("V_primeFinal =", V_prime)
            print("othooooo ", orth(V_prime))
            print("V_f", V_f)
            V_f = orth( V_prime, 1e-6)
            print("--------V_F",V_f)
            quit = (r == np.linalg.matrix_rank(V_f))
            print("r", r)
        
        Reach_mat = V_f
        print("Reach_mat", Reach_mat)
        reduced_order = Reach_mat.shape[1]

        Ar = np.zeros((self.np * reduced_order, reduced_order))
        Br = Reach_mat.T @ Bnum
        Cr = Cnum @ Reach_mat
        x0r = Reach_mat.T @ x0

        for q in range(self.np):
            Aq = Anum[q * self.nx:(q + 1) * self.nx, :]
            Ar[q * reduced_order:(q + 1) * reduced_order, :] = Reach_mat.T @ Aq @ Reach_mat

        return Reach_mat, reduced_order, Ar, Br, Cr, x0r


