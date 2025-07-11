import numpy as np
from src.lpv.dLPV import dLPV
from src.lpv.asLPV import asLPV

from src.lpv.mode.strategy_psi import strategy_psi
from src.lpv.mode.strategy_Myu import strategy_Myu

def wrap_func(x):
    """
    Function that wraps x in a list given that x is an int
    Used in flatten_w and for concatenation

    Parameters :
        x : Any given argument

    """
    if isinstance(x,int):
        return [x]
    else:
        return x

def flatten_w(w):
    """
    Functions that flattens w from a list of doubles and empty lists to an only int list

    Example :
        flatten_w([1.0,[],2.0,0]) => [1,2,0]

    Parameters :
        w : the set of words
    
    """

    sig_j,v_j,sig,u_i = w[0],w[1],w[2],w[3]

    sig_j = wrap_func(sig_j)
    sig = wrap_func(sig)
    v_j = wrap_func(v_j)
    u_i = wrap_func(u_i)

    w = [np.array(sig_j),np.array(v_j),np.array(sig),np.array(u_i)]
    
    w = np.concatenate(w)
    w = w.tolist()
    w = [int(i) for i in w]  
    
    return w


class HoKalmanIdentifier:
    """
    Class implementing the Ho-Kalman and the TrueHokalmanBase algorithms for identification of Linear Parameter Varying.
     
    Attributs
    
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

    Methods :
    
        TrueHoKalamanBase()
    
        switchMode()

        deduce_w_from_base_Hab()

        deduce_w_from_base_Habk()

        deduce_w_from_base_Hak()

        deduce_w_from_base_Hkb()

        Compute_Hab()

        Compute_Habk()

        Compute_Hak()

        Compute_Hkb()

        identify(Hab, Habk_list, Hak_list, Hkb_list)
            Perform the Ho-Kalman identification step.
       
        Seperate_Bsig()

        Compute_Tsig()

        compute_K_Q()

    """
    def __init__(self,A,B,C,D,K,Q,base):
        self.base = base
        self.alpha = base[0]
        self.beta = base[1]

        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.K = K
        self.Q = Q
        self.F = np.eye(1)
        self.mode = strategy_psi()

        # Defining the stochastic part to calculate G_true
        self.as_intern_sys = asLPV(A,C,K,self.F)

    def switchMode(self):
        """
        Switches the current strategy to the other :
            - if the current strategy is strategy_psi => strategy_Myu
            - if the curent strategy is strategy_Myu => strategy_Psi
        """
        if (isinstance(self.mode, strategy_psi)):
            self.mode = strategy_Myu()
        else:
            self.mode = strategy_Psi()

    def deduce_w_from_base_Hab(self,i,j):
        """
        Deduces w (more precisly omega), from the base attribut, used for Hankel matrices Hab
        
            i : int
                Arbitrary index

            j : int
                Arbitrary index

        Returns : A Tuple containing :  
            w the set of words
            k_i, l_j indexs

        """

        sig = []

        sig_j = self.beta[j,0]
        v_j = self.beta[j,1]
        u_i = self.alpha[i,1]
        k_i = self.alpha[i,2]
        l_j = self.beta[i,2]
          
        w = [sig_j,v_j,sig,u_i]
        
        w = flatten_w(w)

        return w,k_i,l_j

    def deduce_w_from_base_Habk(self,i,j,sig):
        """
        Deduces w (more precisly omega), from the base attribut, used for Hankel matrices Hab
        
            i : int
                Arbitrary index

            j : int
                Arbitrary index

        Returns : A Tuple containing :  
            w the set of words
            k_i, l_j indexs

        """
        
        sig_j = self.beta[j,0]
        v_j = self.beta[j,1]
        u_i = self.alpha[i,1]
        k_i = self.alpha[i,2]
        l_j = self.beta[i,2]

        w = [sig_j,v_j,sig,u_i]
        
        w = flatten_w(w)

        return w,k_i,l_j

    def deduce_w_from_base_Hak(self,i,j,sig):
        """
        Deduces w (more precisly omega), from the base attribut, used for Hankel matrices Hak
        
            i : int
                Arbitrary index

            j : int
                Arbitrary index

        Returns : A Tuple containing :  
            w the set of words 
            k_i, l_j indexs

        """
        sig_j = []
        v_j = []
        u_i = self.alpha[i,1]

        k_i = self.alpha[i,2]
        l_j = j
        
        w = [sig_j,v_j,sig,u_i]
        
        w = flatten_w(w)
        
        return w,k_i,l_j

    def deduce_w_from_base_Hkb(self,i,j):
        """
        Deduces w (more precisly omega), from the base attribut, used for Hankel matrices Hak
        
            i : int
                Arbitrary index

            j : int
                Arbitrary index

        Returns : A Tuple containing :  
            w : the set of words 
            k_i, l_j : indexs

        """
        sig_j = self.beta[j,0]
        v_j = self.beta[j,1]
        sig = []
        u_i = []
        
        k_i = i
        l_j = self.beta[j,2]

        w = [sig_j,v_j,sig,u_i]
        
        w = flatten_w(w)

        return w,k_i,l_j
        

    def compute_Hab(self,psig,G):
        """
        Computes the first sub-hankel matrice used for an LPV-SS model
        
        Parameters :
            psig : np.ndarray [np,1]
                Probability weights for each mode.

            G : np.ndarray 
                The matrix that is computed thanks to compute_Gi

        Returns :
            Hab : np.ndarray
                2D array that represents that first sub-hankel matrix
        
        """
        sz_alpha,sz_beta = self.alpha.shape[0],self.beta.shape[0]

        Hab = np.zeros((sz_alpha,sz_beta))
        for i in range(0,sz_alpha):
            for j in range(0,sz_beta):
                w,k_i,l_j = self.deduce_w_from_base_Hab(i,j)
                params = [w,self.A,self.B,self.C,self.D,G,psig]
                M = self.mode.build_M(params)
                Hab[i,j] = M[k_i,l_j]
        return np.round(Hab,4)


    def compute_Habk(self,psig,G):
        """
        Computes the second sub-hankel matrice used for an LPV-SS model
        
        Parameters :
            psig : np.ndarray [np,1]
                Probability weights for each mode.

            G : np.ndarray 
                The matrix that is computed thanks to compute_Gi

        Returns :
            Habk : np.ndarray
                3D array that represents the second sub-hankel matrix
        
        """
        sz_alpha, sz_beta = self.alpha.shape[0],self.beta.shape[0]
        
        np_ = self.A.shape[0]
        
        Habk = np.zeros((np_, sz_alpha, sz_beta))
        for sig in range(0,np_):    
            Habqtmp = np.zeros((sz_alpha,sz_beta))
            for i in range(0,sz_alpha):
                for j in range(0,sz_beta):
                    w,k_i,l_j = self.deduce_w_from_base_Habk(i,j,sig)
                    params = [w,self.A,self.B,self.C,self.D,G,psig]
                    M = self.mode.build_M(params)
                    Habqtmp[i,j] = M[k_i,l_j]
            Habk[sig,:,:] = Habqtmp
        return np.round(Habk,4)


    def compute_Hak(self,psig,G):
        """    
        Computes the third sub-hankel matrice used for an LPV-SS model
        
        Parameters :
            psig : np.ndarray [np,1]
                Probability weights for each mode.

            G : np.ndarray 
                The matrix that is computed thanks to compute_Gi

        Returns :
            Hak : np.ndarray
                3D array reprensting the third sub-hankel matrix

        
        """
        nu = self.B.shape[2] 
        ny = self.C.shape[0]
        np_ = self.A.shape[0]
        sz_alpha = self.alpha.shape[0]
        
        dim = self.mode.search_hak_row_dim(nu,ny)

        Hak = np.zeros((np_,sz_alpha,dim))
        for sig in range(0,np_):
            Hakqtmp = np.zeros((sz_alpha,dim))
            for i in range(0,sz_alpha):
                for j in range(0,dim):
                    w,k_i,l_j = self.deduce_w_from_base_Hak(i,j,sig)
                    params = [w,self.A,self.B,self.C,self.D,G,psig]
                    M = self.mode.build_M(params)
                    Hakqtmp[i,j] = M[k_i,l_j]

            Hak[sig,:,:] = Hakqtmp

        return np.round(Hak,4) 

    def compute_Hkb(self,psig,G):
        """
        Computes the last sub-hankel matrice used for an LPV-SS model
        
        Parameters :
            psig : np.ndarray [np,1]
                Probability weights for each mode.

            G : np.ndarray 
                The matrix that is computed thanks to compute_Gi 
        
        Returns :
            Hkb : np.ndarray
                3D array representing the last sub-hankel matrix

        """
        ny = self.C.shape[0]
        np_c = 1

        sz_beta = self.beta.shape[0]
        
        Hkb = np.zeros((np_c,ny,sz_beta))
        for sig in range(0,np_c):
            Hkbqtmp = np.zeros((ny,sz_beta))
            for i in range(0,ny):
                for j in range(0,sz_beta):
                    w,k_i,l_j = self.deduce_w_from_base_Hkb(i,j)
                    params = [w,self.A,self.B,self.C,self.D,G,psig]
                    M = self.mode.build_M(params)
                    Hkbqtmp[i,j] = M[k_i,l_j]
            Hkb[sig,:,:] = Hkbqtmp
        return np.round(Hkb,4)


    def TrueHoKalmanBase(self,psig):
        """
        Computes the Sub-Hankel matrices used for an LPV-SS model
        (Hab,Habk,Hak,Hkb)

        Parameters :
            psig : np.ndarray
                Probability weights for each mode

        Returns :
            the Sub-Hankel matrices : (Hab,Habk,Hak,Hkb)

        """

        P = self.as_intern_sys.compute_Pi(psig,self.Q)
        G = self.as_intern_sys.compute_Gi(psig,self.Q,P)

        Hab = self.compute_Hab(psig,G)
        Habk = self.compute_Habk(psig,G)
        Hak = self.compute_Hak(psig,G)
        Hkb = self.compute_Hkb(psig,G)

        return (Hab,Habk,Hak,Hkb)

    def identify(self,Hab, Habk, Hak, Hkb):
        """
        Identify system matrices A, B, C from Hankel matrices.

        Parameters :
        
            Hab : np.ndarray, shape (sz_alpha, sz_beta)
                Base Hankel matrix H_M (indexed by past and future words).

            Habk : np.ndarray, each shape (sz_alpha, sz_beta)
                3D matrix containing shifted Hankel matrices H_M^shifted, one per mode/scheduling signal sigma.
        
            Hak : list of np.ndarray, each shape (sz_alpha, nu)
                3D matrix containing Hankel matrices related to inputs, one per mode/sigma.
        
            Hkb : list of np.ndarray, each shape (ny, nx)
                3D matrix containing Hankel matrices related to outputs, one per mode/sigma.

        Returns :
        
            Asig : np.ndarray, shape (nx, nx, np)
                   Array of state transition matrices A.
        
            Bsig : np.ndarray, shape (nx, nu, np)
                   Array of input matrices B.
        
            Csig : np.ndarray, shape (ny, nx, np_c)
                   Array of output matrices C.
        
        """
        
        nx = Hab.shape[0]

        nu = Hak[0,:,:].shape[1]
        ny = Hkb[0,:,:].shape[0]
        np_ = self.A.shape[0]
        np_c = 1 # the attribut C has 1 of depth

        A = np.zeros((np_, nx, nx))
        B = np.zeros((np_, nx, nu))
        C = np.zeros((np_c, ny, nx))

        Hab_inv = np.linalg.pinv(Hab)  

        for i in range(np_):
            A[i, :, :] = Hab_inv @ Habk[i,:,:]
            eigvals = np.linalg.eigvals(A[i, :, :])
            if np.any(np.abs(eigvals) > 1):
                print(f"Warning: A matrix at index {i} is unstable (eigenvalues outside unit circle)")

            B[i, :, :] = Hab_inv @ Hak[i,:,:]
        
        for i in range(np_c):
            C[i,:,:] = Hkb[i,:,:]

        return dLPV(A,C,B,self.D)
    
    def seperate_Bsig(self,Bsig):
        """
        Seperates the matrix Bsig into two matrices such as : Bsig = [Bi,Gi]
        
        Paramaters :
            Bsig : np.ndarray
                The deduced B matrix (inside the dLPV) thanks to the identify function 

        Returns :
            (Bi,Gi) : (np.ndarray,np.ndarray)
        """
        nu = self.D.shape[1]

        Gi = Bsig[:,:,nu:]
        Bi = Bsig[:,:,:nu]

        return (Bi,Gi)

    def compute_Tsig(self,Asig,Csig,Bsig,psig):
        """
        Computes T_sig

        Parameters :
            Asig : np.ndarray
                The deduced A matrix (inside the dLPV) thanks to the identify function

            Csig : np.ndarray
                The deduced C matrix (inside the dLPV) thanks to the identify function

            Bsig : np.ndarray
                The deduced B matrix (inside the dLPV) thanks to the identify function

            psig : np.ndarray
                Parameters weights for each mode

        """
        
        print("Bsig")
        print(Bsig)
        print(Bsig.shape)

        Bi,Gi = self.seperate_Bsig(Bsig)

        print("check Bi")
        print(Bi.shape)
        print(Bi)

        #C = Csig[0,:,:] #Transform 3D Csig to a 2D Matrix (As used in asLPV)
        
        print("Check Gi")
        print(Gi.shape)

        print(Gi)
        
        asLPV_sys = asLPV(Asig,Csig,Gi,self.F)

        np_ = Asig.shape[0]

        P_true = asLPV_sys.compute_Pi(psig,self.Q)
        G_true = asLPV_sys.compute_Gi(psig,self.Q,P_true)

        T_sig_true = np.zeros((Asig.shape[0],Csig.shape[0],Csig.shape[0]))

        for i in range(np_):
            T_sig_true[i,:,:] = (1/psig[i,0])*(Csig @ P_true[i,:,:] @ Csig.T + self.F @ self.Q[i,:,:] @ self.F.T)

        return T_sig_true


    def compute_K_Q(self,Asig,Bsig,Csig,psig):
        """
        Parameters :
            Asig : np.ndarray
                The deduced A matrix (inside the dLPV) thanks to the identify function

            Csig : np.ndarray
                The deduced C matrix (inside the dLPV) thanks to the identify function

            Bsig : np.ndarray
                The deduced B matrix (inside the dLPV) thanks to the identify function
            
            psig : np.ndarray
                Parameters weights for each mode
            
            T_sig : np.ndarray
                Computed thanks to compute_Tsig

        Returns :
            
            Q_old : ndarray
            
            K_old : ndarray


        """
        C = Csig[0,:,:]

        _,Gi = self.seperate_Bsig(Bsig)

        dLPV_sys = dLPV(Asig,C,Gi,self.D)
        
        T_sig = self.compute_Tsig(Asig,C,Bsig,psig)

        (_,Q_old,K_old) = dLPV_sys.Recursion(T_sig,psig)

        return Q_old,K_old

        
