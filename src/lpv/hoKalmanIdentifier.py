import numpy as np
from src.lpv.dLPV import dLPV

from src.lpv.modeStrategy.strategy_psi import strategy_psi
from src.lpv.modeStrategy.strategy_Myu import strategy_Myu

class HoKalmanIdentifier:
    """
    Class implementing the Ho-Kalman and the TrueHokalmanBase algorithms for identification of Linear Parameter Varying.
     
    Attributs
    --------
    A : np.ndarray
        3D array representing the state transition matrices.

    B : np.ndarray
        3D array representing the input matrices.
    
    C : np.ndarray
        2D array representing the output matrix.
    
    D : np.ndarray
        2D array representing the feedthrough matrix.

    base : Tuple containing two np.ndarray such as base = (alpha,beta)
        alpha : word index set
        beta : word index set

    mode : An instance of the modeStrategy class
        it will be used for changing how M(y,u) is calculated.

    Methods
    -------
    TrueHoKalamanBase()

    Compute_Hab()

    Compute_Habk()

    Compute_Hak()

    Compute_Hkb()

    identify(Hab, Habk_list, Hak_list, Hkb_list)
        Perform the Ho-Kalman identification step.
    
    """
    def __init__(A,B,C,D,base):
        self.base = base
        self.alpha = base[0]
        self.beta = base[1]

        self.A = A
        self.B = B
        self.C = C
        self.D = D
        mode = 

    def deduce_w_from_base_Hab_Habk(i,j):
        """
        Deduces w (more precisly omega), from the base attribut, used for Hankel matrices Habk and Habk
        
        i : int
            Arbitrary index

        j : int
            Arbitrary index

        Returns :
            w :

            k_i :

            l_j :

        """
        sig = []
        sig_j = beta[j,0]
        v_j = beta[j,1]
        u_i = alpha[i,0]
        k_i = alpha[i,2]
        l_j = beta[i,2]

        w = [sig_j,v_j,sig,u_i]
        return w,k_i,l_j

    def deduce_w_from_base_Hak(i,j):
        """
        Deduces w (more precisly omega), from the base attribut, used for Hankel matrices Hak
        
        i : int
            Arbitrary index

        j : int
            Arbitrary index

        Returns :
            w :
            k_i :
            l_j :

        """
        sig_j = []
        v_j = []
        u_i = alpha[i,1]

        k_i = alpha[i,2]
        l_j = j
        
        w = [sig_j,v_j,u_i]
        return w,k_i,l_j

    def deduce_w_from_base_Hkb(i,j):
        """
        Deduces w (more precisly omega), from the base attribut, used for Hankel matrices Hak
        
        i : int
            Arbitrary index

        j : int
            Arbitrary index

        """
        sig_j = beta[j,0]
        v_j = beta[j,1]
        u_i = []
        
        k_i = i
        l_j = beta[j,2]

        w = [sig_j,v_j,u_i]
        return w,k_i,l_j
        

    def compute_Hab():
        """
        Computes the first sub-hankel matrice used for an LPV-SS model
        """
        sz_alpha,sz_beta = alpha.shape[0],beta.shape[0]

        Hab = np.zeros(sz_alpha,sz_beta)
        for i in range(0,sz_alpha):
            for j in range(0,sz_beta):
                w,k_i,l_j = deduce_w_from_base_Hab_Habk(i,j)
                #M = mode.buildM() discuss about G matrix
                Hab[i,j] = M[k_i,l_j]

        return Hab


    def compute_Habk():
        """
        Computes the second sub-hankel matrice used for an LPV-SS model
        """

    def compute_Hak():
        """    
        Computes the third sub-hankel matrice used for an LPV-SS model
        """

    def compute_Hkb():
        """
        Computes the last sub-hankel matrice used for an LPV-SS model
        """

    def TrueHoKalmanBase():
        """
        Computes the Sub-Hankel matrices used for an LPV-SS model
        (Hab,Habk,Hak,Hkb)

        Returns :
        the Sub-Hankel matrices : (Hab,Habk,Hak,Hkb)

        """
       Hab = self.compute_Hab()
       Habk = self.compute_Habk()
       Hak = self.compute_Hak()
       Hkb = self.compute_Hkb()

       return (Hab,Habk,Hak,Hkb)
    
    @staticmethod
    def identify(Hab, Habk, Hak, Hkb):
        """
        Identify system matrices A, B, C from Hankel matrices.

        Parameters
        ----------
        Hab : np.ndarray, shape (nx, nx)
              Base Hankel matrix H_M (indexed by past and future words).
        
        Habk : list of np.ndarray, each shape (nx, nx)
               List of shifted Hankel matrices H_M^shifted, one per mode/scheduling signal sigma.
        
        Hak : list of np.ndarray, each shape (nx, nu)
              List of Hankel matrices related to inputs, one per mode/sigma.
        
        Hkb : list of np.ndarray, each shape (ny, nx)
              List of Hankel matrices related to outputs, one per mode/sigma.

        Returns
        -------
        Asig : np.ndarray, shape (nx, nx, np)
               Array of state transition matrices A.
        
        Bsig : np.ndarray, shape (nx, nu, np)
               Array of input matrices B.
        
        Csig : np.ndarray, shape (ny, nx, np_c)
               Array of output matrices C.
        
        """
        
        nx = Hab.shape[0]
        nu = Hak[0].shape[1]
        ny = Hkb[0].shape[0]
        np_ = len(Habk)
        np_c = len(Hkb)

        A = np.zeros((nx, nx, np_))
        B = np.zeros((nx, nu, np_))
        C = np.zeros((ny, nx, np_c))

        Hab_inv = np.linalg.pinv(Hab)  

        for i in range(np_):
            A[:, :, i] = Hab_inv @ Habk[i]
            eigvals = np.linalg.eigvals(A[:, :, i])
            if np.any(np.abs(eigvals) > 1):
                print(f"Warning: A matrix at index {i} is unstable (eigenvalues outside unit circle)")

            B[:, :, i] = Hab_inv @ Hak[i]

        for i in range(np_c):
            C[:, :, i] = Hkb[i]

        return A, B, C
    
    @staticmethod
    def Hokalman_to_dLPV(A,B,C,D,A_hok, B_hok, C_hok):
        D_hok = psi_uy_true([], A, B, C, D)
        B_hok = B_hok[:, :nu, :]  
        return dLPV(A_hok, C_hok, B_hok, D_hok)
    
    
        
