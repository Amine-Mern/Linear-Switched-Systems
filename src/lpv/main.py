import numpy as np
from asLPV import asLPV

def calculate_v(ny, Ntot):
    return np.random.randn(ny,Ntot)
    
def calculate_psig(p):
    np_, Ntot = p.shape
    psig = np.zeros((np_, 1))
    for i in range(np_):
        psig[i, 0] = np.var(p[i, :]) + np.mean(p[i, :])**2
    return psig

def check_dimensions(A, C, K, F, p):
    nx, _, n_sig = A.shape
    ny, nx_c = C.shape
    nx_k, nw, n_sig_k = K.shape
    ny_f, nw_f = F.shape
    np_, Ntot = p.shape

    assert nx == nx_c, f"Dimension mismatch: A has {nx} states, but C has {nx_c} columns"
    assert nx_k == nx, f"K should have {nx} rows (states), but has {nx_k}"
    assert n_sig == n_sig_k, f"A and K must have the same number of scheduling points: {n_sig} vs {n_sig_k}"
    assert ny == ny_f, f"C and F must have same number of outputs: {ny} vs {ny_f}"
    assert nw == nw_f, f"K and F must have same number of noise inputs: {nw} vs {nw_f}"
    return True 


def main(A, C, K, F, p, x0=None):
    check_dimensions(A, C, K, F, p)
    nx = A.shape[0]
    ny = C.shape[0]
    Ntot = p.shape[1]

    if x0 is None:
        x0 = np.zeros((nx, 1))
    
    v = calculate_v(ny, Ntot)
    psig = calculate_psig(p)
    
    system = asLPV(A, C, K, F)
    
    as_min_system, Qmin = system.stochMinimize(v, p, psig)
    
    # Résultat
    print("\n--- Minimised system ---")
    print("A:\n", as_min_system.A)
    print("C:\n", as_min_system.C)
    print("K:\n", as_min_system.K)
    print("F:\n", as_min_system.F)
    print("Qmin:\n", Qmin)

    return as_min_system, Qmin
        
        
if __name__ == "__main__":
#     A = np.random.randn(2, 2, 2)
#     K = np.random.randn(2, 1, 2)
#     C = np.random.randn(1, 2)
#     F = np.eye(1)
#     p = np.random.randn(2, 100)
    A = np.array([
        [[0.5, 0.1],
         [0.0, 0.3]],
        [[0.2, 0.05],
         [0.0, 0.4]]
    ]).transpose(1, 2, 0)  # (2, 2, 2)

    K = np.random.randn(2, 1, 2) * 0.05
    C = np.array([[1.0, 0.0]])
    F = np.eye(1)
    p = np.random.randn(2, 100)
    main(A, C, K, F, p)
