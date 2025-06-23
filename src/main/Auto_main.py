import numpy as np
from src.lpv.asLPV import asLPV

def calculate_v(ny, Ntot):
    return np.random.randn(ny,Ntot)

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


def asLPVGen(nx, ny, nv, np_, psig):
    """
    Generates a stable asLPV system (innovation form + autonomous stability).
    
    Parameters:
        - nx : number of states
        - ny : number of outputs
        - nv : number of noise inputs
        - np_ : number of scheduling points
        - psig : probability distribution over the scheduling points
    
    Returns:
        system : a stable asLPV
    """
    out = False
    while not out:
        A = np.random.randn(nx, nx, np_)
        K = np.random.randn(nx, nv, np_) * 0.1
        C = np.random.randn(ny, nx)
        F = np.eye(ny)
        system = asLPV(A, C, K, F)

        form_stable = system.isStablyInvertable(psig)
        autonomous_stable = system.is_A_matrix_stable()

        if form_stable and autonomous_stable:
            out = True
    print("A =")
    for i in range(np_):
        print(f"A[:,:,{i}] =\n{A[:, :, i]}")
    print("\nK =")
    for i in range(np_):
        print(f"K[:,:,{i}] =\n{K[:, :, i]}")
    print("\nC =\n", C)   

    return system


def main(nx, ny, nv, np_):
    psig = np.ones((np_, 1)) / np_ 
    system = asLPVGen(nx, ny, nv, np_, psig)
    p = np.random.randn(np_, 100)
    try:
        check_dimensions(system.A, system.C, system.K, system.F, p)
        print("✅ Dimensions are consistent.")
    except AssertionError as e:
        print("❌ Dimension check failed:", e)

    print("✅ Innovation Form Stability:", system.isStablyInvertable(psig))
    print("✅ Autonomous Stability:", system.is_A_matrix_stable())
    
    Ntot = p.shape[1]
    v = calculate_v(ny, Ntot)
    as_min_system, Qmin = system.stochMinimize(v, p, psig)
    
    print("\n--- Minimised system ---")
    print("A:\n", as_min_system.A)
    print("C:\n", as_min_system.C)
    print("K:\n", as_min_system.K)
    print("F:\n", as_min_system.F)
    print("Qmin:\n", Qmin)
    
    print("✅  Stably invertible:", as_min_system.isStablyInvertable(psig))
    
if __name__ == "__main__":
    main(3,1,1,2)

