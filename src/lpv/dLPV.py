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
    
    simulate_y()
    
    minimizeDLPV()
    
    reach_reduction()
    
    obsRedLPV()
    
    isIsomorphic()
    
    Recursion()
    """
    
    def __init__(self, A, C, B, D):
        super().__init__(A, C, B, D, K=None, F=None)
        
    def simulate_y(self, u, v, p, Ntot):
        """
        Simulate the dLPV system output
        
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
            - yif : ndarray
                    Output input free
            - x : ndarray
                  State trajectory
        """
        nx, ny, np_ = self.nx, self.ny, self.np
        x = np.zeros((nx, Ntot))
        y = np.zeros((ny, Ntot))
        yif = np.zeros((ny, Ntot))
        
        for k in range (Ntot-1):
            for i in range(np_):
                x[:, k+1] += (self.A[:, :, i] @ x[:, k] + self.B[:, :, i] @ u[:, k]) * p[i, k] 
            y[:, k] = self.C @ x[:, k] + self.D @ u[:, k]
            yif[:, k] = self.C @ x[:, k]
        return y, yif, x
    
#     @staticmethod
#     def __get_tolerance(s, max_size_A):
#         """
#         Compute tolerance for singular values comparison used in orthogonalization.
# 
#         Parameters:
#             s : np.ndarray
#                 Array of singular values.
#             max_size_A : int
#                          Maximum dimension size of the matrix for tolerance scaling.
# 
#         Returns:
#                 float
#                 Tolerance value for determining numerical rank.
#         """
#         if s.size == 0:
#             return 0.0
#         if not np.all(np.isfinite(s)):
#             return np.finfo(s.dtype).max # realmax
#         return max_size_A * np.spacing(np.max(s))
#     
#     @staticmethod
#     def svd_matlab_style(A):
#         U, s, Vh = np.linalg.svd(A, full_matrices=False)
#         
#         for i in range(U.shape[1]):
#             col = U[:, i]
#             first_nonzero = np.flatnonzero(col)
#             if first_nonzero.size > 0 and col[first_nonzero[0]] < 0:
#                 U[:, i] *= -1
#                 Vh[i, :] *= -1 
# 
#         return U, s, Vh
#     
#     @staticmethod
#     def __orth(self, A, tol=None):
#         """
#         Compute an orthonormal basis for the range of A.
#     
#         Parameters:
#             A : np.ndarray
#                 Input matrix.
#             tol : float, optional
#                 Tolerance for small singular values. If None, it will be computed.
#     
#         Returns:
#             Q : np.ndarray
#                 Orthonormal basis for the range of A. That is, Q'*Q = I, the columns of Q span the same space as 
#                 the columns of A, and the number of columns of Q is the rank of A.
#         """
#         U, s, Vh = self.svd_matlab_style(A)
#         print("/////////////////U = ", U)
#         V = Vh.T
#         Vh = V
#         
#         if s.size>0 and (np.isnan(s[0] or np.isinf(s[0]))):
#             raise ValueError("Input matrix contains Nan or Inf values.")
#         
#         if tol is None:
#             tol = self.__get_tolerance(s, max(A.shape))
#         else:
#             if not (np.isscalar(tol) and isinstance(tol, (float, int))):
#                 raise ValueError("Tolerance must be a real scalar.")
#             
#         rank = np.sum(s>tol)
#         Q = U[:, :rank]
#         return Q
    
    
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
        
        V_f = orth( B_hat)
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
            V_f = orth(V_prime)
            print("--------V_F",V_f)
            quit = (r == np.linalg.matrix_rank(V_f))
            print("r", r)
        
        Reach_mat = V_f
        print("Reach_mat", Reach_mat)
        reduced_order = Reach_mat.shape[1]

        Ar = np.zeros((reduced_order, reduced_order, self.np))
        Br = np.zeros((reduced_order, self.nu, self.np))
        
        for i in range(self.np):
            Ar[:, :, i] = Reach_mat.T @ self.A[:, :, i] @ Reach_mat
            Br[:, :, i] = Reach_mat.T @ self.B[:, :, i]

        Cr = self.C @ Reach_mat
        x0r = Reach_mat.T @ x0
        
        for i in range(self.np):
            print(f"Ar[:,:,{i}]:\n", Ar[:, :, i])
        for i in range(self.np):
            print(f"Br[:,:,{i}]:\n", Br[:, :, i])
        print("Cr = ",  Cr)
        print("x0r = ", x0r)
        print("reach_mat = ", Reach_mat)
        print("r = " ,reduced_order)
        
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
        """
        print("-----------------obssss startedd---------------")
        Cnum = np.vstack([self.C for _ in range(self.np)])
        x0 = x0.reshape(-1,1)
        print("Cnum =" ,Cnum)
        Wf = orth(Cnum.T)
        Wf = Wf
        print("Wf = " , Wf)
        r = np.linalg.matrix_rank(Wf)
        print("r", r)

        quit = False
        while not quit:
            Wprime = Cnum.T
            for i in range(self.np):
                Wprime = np.hstack((Wprime, self.A[:, :, i].T @ Wf))
                print("W_prime = ", Wprime, i)
            new_Wf = orth(Wprime)
            print("new_Wf = ", new_Wf, i)
            quit = new_Wf.shape[1] == r
            r = new_Wf.shape[1]
            Wf = new_Wf

        reduced_order = r

        Ar = np.zeros((reduced_order, reduced_order, self.np))
        Br = np.zeros((reduced_order, self.B.shape[1], self.np))
        for i in range(self.np):
            Ar[:, :, i] = Wf.T @ self.A[:, :, i] @ Wf
            Br[:, :, i] = Wf.T @ self.B[:, :, i]

        Cr = self.C @ Wf
        x0r = Wf.T @ x0

        Obs_mat = Wf
        for i in range(self.np):
            print(f"Ao[:,:,{i}]:\n", Ar[:, :, i])
        for i in range(self.np):
            print(f"Bo[:,:,{i}]:\n", Br[:, :, i])
        print("Co = ",  Cr)
        print("x0r = ", x0r)
        print("Obs_mat = ", Obs_mat)
        print("r = " ,reduced_order)
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
        """
        tol = 1e-10
        if not isinstance(other, dLPV):
            raise ValueError("Other system must be an instance of dLPV.")
        if self.nx != other.nx or self.np != other.np:
            return False
        
        Anum3 = np.zeros((2 * self.nx, 2 * self.nx, self.np))
        Bnum3 = np.zeros((2 * self.nx, self.nu, self.np))
        
        
        # Create merged system (combined observability test)
        for i in range(self.np):
            Anum3[:, :, i] = np.block([
                [self.A[:, :, i], np.zeros((self.nx, self.nx))],
                [np.zeros((self.nx, self.nx)), other.A[:, :, i]]
            ])
            Bnum3[:, :, i] = np.vstack([
                self.B[:, :, i],
                other.B[:, :, i]
            ])
        print("Anum3 = ", Anum3)
        print("Bnum3 = ", Bnum3)
        Cnum3 = np.hstack([self.C, other.C])
        print("Cnum3 = ", Cnum3)
        Dnum3 = np.hstack([self.D, other.D])

        x03 = np.vstack([x0.reshape(-1, 1), x0_other.reshape(-1, 1)])
        print("x03 = ", x03)
        
        merged_dLPV = dLPV(Anum3, Cnum3, Bnum3,self.D)
        obs_mat, obs_rank, Ao, Bo, Co, x0o = merged_dLPV.obs_reduction(x03)

        S = obs_mat.T
        T1 = S[:, :self.nx]
        T2 = S[:, self.nx:]

        try:
            T = np.linalg.inv(T2) @ T1
        except np.linalg.LinAlgError:
            print("T is not inversible therefore the dLPVs are not isomorphic")
            return False  

        for i in range(self.np):
            Ai1 = self.A[:, :, i]
            Ai2 = other.A[:, :, i]
            if np.linalg.norm(T @ Ai1 - Ai2 @ T) > tol:
                return False

            Bi1 = self.B[:, :, i]
            Bi2 = other.B[:, :, i]
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
            """
            A = self.A
            G = self.G
            C = self.C

            nx = self.nx
            ny = self.ny
            np_ = self.np

            Pold = np.zeros((nx, nx, np_))
            Qold = np.zeros((ny, ny, np_))
            Kold = np.zeros((nx, ny, np_))
            max_err = np.ones(np_)

            while np.any(max_err > 1e-8):

                for sig in range(np_):
                    Qold[:, :, sig] = psig[sig, 0] * T_sig[:, :, sig] - C @ Pold[:, :, sig] @ C.T
                    
                    invQ = np.linalg.inv(Qold[:, :, sig])
                    sqrt_psig = np.sqrt(psig[sig, 0])
                    inv_sqrt_psig = 1.0 / sqrt_psig
                    Kold[:, :, sig] = (sqrt_psig * G[:, :, sig] -
                                       inv_sqrt_psig * A[:, :, sig] @ Pold[:, :, sig] @ C.T) @ invQ
                    Pnew = np.zeros_like(Pold)
                for sig in range(np_):
                    for sig1 in range(np_):
                        term1 = (1 / psig[sig1, 0]) * A[:, :, sig1] @ Pold[:, :, sig1] @ A[:, :, sig1].T
                        term2 = Kold[:, :, sig1] @ Qold[:, :, sig1] @ Kold[:, :, sig1].T
                        Pnew[:, :, sig] += psig[sig, 0] * (term1 + term2)

                for i in range(np_):
                    num = np.linalg.norm(Pnew[:, :, i] - Pold[:, :, i])
                    den = np.linalg.norm(Pold[:, :, i]) + 0.1
                    max_err[i] = num / den

                Pold = np.copy(Pnew)

            return Pold, Qold, Kold

