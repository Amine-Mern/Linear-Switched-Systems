from LPV import LPV
import numpy as np

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
    
    def __init__(self, A, C, B, D):
        super().__init__(A, C, B, D, K=None, F=None)
    
    @staticmethod
    def __get_tolerance(s, max_size_A):
        """
        Compute tolerance for singular values comparison used in orthogonalization.

        Parameters:
            s : np.ndarray
                Array of singular values.
            max_size_A : int
                         Maximum dimension size of the matrix for tolerance scaling.

        Returns:
                float
                Tolerance value for determining numerical rank.
        """
        if s.size == 0:
            return 0.0
        if not np.all(np.isfinite(s)):
            return np.finfo(s.dtype).max # realmax
        return max_size_A * np.spacing(np.max(s))
    
    @staticmethod
    def __orth(self, A, tol=None):
        """
        Compute an orthonormal basis for the range of A.
    
        Parameters:
            A : np.ndarray
                Input matrix.
            tol : float, optional
                Tolerance for small singular values. If None, it will be computed.
    
        Returns:
            Q : np.ndarray
                Orthonormal basis for the range of A. That is, Q'*Q = I, the columns of Q span the same space as 
                the columns of A, and the number of columns of Q is the rank of A.
        """
        U, s, Vh = np.linalg.svd(A, full_matrices=False)
        
        if s.size>0 and (np.isnan(s[0] or np.isinf(s[0]))):
            raise ValueError("Input matrix contains Nan or Inf values.")
        
        if tol is None:
            tol = self.__get_tolerance(s, max(A.shape))
        else:
            if not (np.isscalar(tol) and isinstance(tol, (float, int))):
                raise ValueError("Tolerance must be a real scalar.")
            
        rank = np.sum(s>tol)
        Q = U[:, :rank]
        return Q
    
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
        Bnum = np.hstack([self.B[:, :, i] for i in range(self.np)])
        Cnum = np.vstack([self.C for _ in range(self.np)])
        x0 = x0.reshape(-1, 1)
        B_hat = np.hstack([x0, Bnum])
        
        V_f = self.__orth(self, B_hat)
        V_0 = V_f.copy()
        
        quit = False
        while not quit :
            r = np.linalg.matrix_rank(V_f)
            V_prime = V_0.copy()
            for j in range(self.np):
                Aj = Anum[j * self.nx:(j + 1) * self.nx, :]
                V_prime = np.hstack([V_prime, Aj @ V_f])
            V_f = self.__orth(self, V_prime)
            quit = (r == np.linalg.matrix_rank(V_f))
        
        Reach_mat = V_f
        reduced_order = Reach_mat.shape[1]

        Ar = np.zeros((self.np * reduced_order, reduced_order))
        Br = Reach_mat.T @ Bnum
        Cr = Cnum @ Reach_mat
        x0r = Reach_mat.T @ x0

        for q in range(self.np):
            Aq = Anum[q * self.nx:(q + 1) * self.nx, :]
            Ar[q * reduced_order:(q + 1) * reduced_order, :] = Reach_mat.T @ Aq @ Reach_mat

        return Reach_mat, reduced_order, Ar, Br, Cr, x0r