import numpy as np
from numpy import eye

def psi_uy_true(w, gam_i, A, B, C, D=None):
    """
    Computes the Markov parameter Ψ_{u,y}(w) for a given word w and output index gam_i.
    
    Parameters:
        w (List[int]): Word (sequence of indices) where w[0] = σ_j, w[1:] = s = switching indices
        gam_i (int): Output mode index
        A (np.ndarray): Array — A matrices for each scheduling mode
        B (np.ndarray): Array — B matrices
        C (np.ndarray): Array — C matrices
        D (np.ndarray): Optional — Direct transmission matrix (ny, nu)
    Returns:
        Psi (np.ndarray): Ψ_{u,y}(w) ∈ R^{ny × nu}
    """
    
    if len(w) == 0:
        if D is None:
            raise ValueError("D must be provided if w is empty.")
        return D
    
    sig_j = w[0]
    s = w[1:]
    nx = A.shape[0]
    As = np.eye(nx)
    for i in s:
        As = A[:, :, i] @ As
    Psi = C[:,:, 1] @ As @ B[:,:, sig_j]
    return Psi


def psi_ys_true(w, gam_i, A, G, C, psig):
    """
    Computes the Markov parameter Ψ_{ys}(w) for a given word w and output index gam_i,
    along with the associated path probability product.
    
    Parameters:
        w (List[int]): Word (sequence of indices), where w[0] = σ_j (noise mode), w[1:] = s (switching indices)
        gam_i (int): Output mode index 
        A (np.ndarray): State transition matrices
        C (np.ndarray): Output matrices
        G (np.ndarray): Noise input matrices
        psig (np.ndarray): Probabilities associated with switching modes.
    Returns:
        Psi (np.ndarray): Ψ_{ys}(w) ∈ R^{ny × ny}, the Markov parameter
        ps (float): Product of switching probabilities along path s
    """
    ny = C.shape[0]

    if len(w) == 0:
        return np.eye(ny), 1.0

    sig_j = w[0]
    s = w[1:]
    nx = A.shape[0]
    As = np.eye(nx)
    ps = 1.0

    for i in s:
        As = A[:, :, i] @ As
        ps *= psig[i]

    Psi = C[:, :, gam_i] @ As @ G[:, :, sig_j] * np.sqrt(ps)
    return Psi, ps

def Myu(sig_j, v_j, sig, u_i, gam_i, A, B, C, D, G, psig):
    """
    Computes Myu(w) = [Ψ_{u,y}(w), Ψ_{ys}(w)] for a composed word w = [σ_j, v_j, σ, u_i].
    Parameters:
        sig_j (int): Input mode index σ_j
        v_j (List[int]): Switching sequence v_j
        sig (List[int]): Switching sequence σ
        u_i (int): Input index u_i
        gam_i (int): Output mode index γ_i
        A, B, C, G (np.ndarray): System matrices 
        D (np.ndarray or None): Direct feedthrough matrix, used if w is empty
        psig (np.ndarray): Probability vector for switching modes
    Returns:
        Myu (np.ndarray): Concatenated Markov vector
    """
    w = [sig_j] + v_j + sig + [u_i]
    Psi_ys, ps = psi_ys_true(w, gam_i, A, G, C, psig)
    
    if len(w) == 0:
        Psi_uy = psi_uy_true(w, gam_i, A, B, C, D)
    else:
        weight = np.sqrt(psig[w[0]] * ps)
        Psi_uy = weight * psi_uy_true(w, gam_i, A, B, C, D)

    Myu = np.hstack([Psi_uy, Psi_ys])
    return Myu

