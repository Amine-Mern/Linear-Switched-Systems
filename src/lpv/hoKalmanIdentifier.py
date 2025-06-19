import numpy as np

class HoKalmanIdentifier:
    """
    Class implementing the Ho-Kalman algorithm for identification of
    Linear Parameter Varying.
    
    The algorithm extracts state-space matrices (A, B, C) from
    block Hankel matrices constructed from Markov parameters.
    
    Methods
    -------
    identify(Hab, Habk_list, Hak_list, Hkb_list)
        Perform the Ho-Kalman identification step.
    
    """
    
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
