import numpy as np

def simulate_y(A, B, K, C, D, F, u, v, p, Ntot):
    """
    Simulates the output of an LPV (Linear Parameter-Varying) system.
    
    Args:
        A, B, C : System matrices
        C, D, F : Output matrices
        u : input to the system
        v : Noise input
        p : Scheduling parameter
        Ntot : Total number of time steps
         
    Returns:
        y : Output with noise
        ynf : Noise-free output
        x : System states
    """
    nx = A.shape[0]
    ny = C.shape[0]
    np_ = p.shape[0]
    
    x = np.zeros((nx, Ntot))
    y = np.zeros((ny, Ntot))
    ynf = np.zeros((ny, Ntot))
    
    for k in range(1,Ntot):
        for i in range(1, np_+1):
            x[:,k+1] += (A[:, :, i] @ x[:, k] + B[:, :, i] @ u[:, k] + K[:, :, i] @ v[:, k]) * p[i, k]
        y[:,k] = C @ x[:,k] + D @ u[:,k] + F @ v[:,k]
        ynf[:,k] = C @ x[:,k]
    
    return y, ynf, x