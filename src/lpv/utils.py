import numpy as np

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
    Psi = C[:,:, gam_i] @ As @ B[:,:, sig_j]
    return Psi