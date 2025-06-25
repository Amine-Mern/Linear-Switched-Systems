import numpy as np

import utils as tools

def deduce_from_base_Hab(base,i,j):
    alpha = base[0]
    beta = base[1]

    sig_j = beta[j,0]
    v_j = beta[j,1]
    u_i = alpha[i,0]
    k_i = alpha[i,2]
    l_j = beta[i,2]

    w = [sig_j,v_j,[],u_i]
    return w,k_i,l_j


def compute_Hab(A,B,C,D,base):
    sz_alpha = base[0].shape[0] #size alpha
    sz_beta = base[1].shape[0] #size beta
    Hab = np.zeros(sz_alpha,sz_beta)
    for i in range(0,sz_alpha):
        for j in range(0,sz_beta):
            w,k_i,l_j = deduce_from_base_Hab(base,i,j)
            M = psi_uy_true(w,A,B,C,D)
            Hab[i,j] = M[k_i,l_j]

    return Hab

def compute_Habk(A,B,C,D,base):

def compute_Hak(A,B,C,D,base):

def compute_Hkb(A,B,C,D,base):

def TrueHoKalmanBase_Det(A,B,C,D,base):
    """
    Computes the Sub-Hankel matrices used for an LPV-SS model
    (Hab,Habk,Hak,Hkb).

    Parameters :

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

    Returns :

    the Sub-Hankel matrices : (Hab,Habk,Hak,Hkb)
    """
    
    #np_ = A.shape[2]
    #np_c = C.shape[2]
    #nu = B.shape[1]
    #ny = C.shape[0]

    Hab = compute_Hab(A,B,C,D,base)
    Habk = compute_Habk(A,B,C,D,base)
    Hak = compute_Hak(A,B,C,D,base)
    Hkb = compute_Hkb(A,B,C,D,base)
