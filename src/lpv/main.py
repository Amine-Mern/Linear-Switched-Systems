import numpy as np
from LPV import LPV
from dLPV import dLPV
from scipy.linalg import orth

def main():

    print("---------- TEST 1 : Equivalent Systems ----------")
    A1 = np.zeros((2, 2, 2))
    B1 = np.zeros((2, 1, 2))
    C1 = np.array([[1.0, 0.0]])
    D1 = np.array([[0.0]])

    A1[:, :, 0] = np.array([[0.8, 0.1],
                            [0.0, 0.5]])
    A1[:, :, 1] = np.array([[0.5, 0.2],
                            [0.1, 0.6]])
    B1[:, :, 0] = np.array([[1.0],
                            [0.0]])
    B1[:, :, 1] = np.array([[0.5],
                            [1.0]])

    T = np.array([[2.0, 0.0],
                  [1.0, 1.0]])
    T_inv = np.linalg.inv(T)

    A2 = np.zeros((2, 2, 2))
    B2 = np.zeros((2, 1, 2))
    for i in range(2):
        A2[:, :, i] = T @ A1[:, :, i] @ T_inv
        B2[:, :, i] = T @ B1[:, :, i]
    C2 = C1 @ T_inv
    D2 = np.array([[0.0]])

    x01 = np.array([0.0, 0.0])
    x02 = np.array([0.0, 0.0])

    sys1 = LPV(A1, C1, B1, D1)
    sys2 = LPV(A2, C2, B2, D2)

    print("Équivalents ?", sys1.isEquivalentTo(sys2, x01, x02), "\n")

    print("\n---------- TEST 2 : Non equivalent systems ----------")
    A1[:, :, 0] = np.array([[0.9, 0.1],
                            [0.0, 0.5]])
    A1[:, :, 1] = np.array([[0.3, 0.0],
                            [0.0, 0.4]])

    B1[:, :, 0] = np.array([[1.0],
                            [0.0]])
    B1[:, :, 1] = np.array([[0.5],
                            [1.0]])

    A2[:, :, 0] = np.array([[0.6, 0.2],
                            [0.1, 0.5]])
    A2[:, :, 1] = np.array([[0.2, 0.1],
                            [0.1, 0.7]])

    B2[:, :, 0] = np.array([[0.8],
                            [0.1]])
    B2[:, :, 1] = np.array([[0.3],
                            [0.9]])

    C2 = C1.copy()
    D2 = D1.copy()

    x01 = np.array([0.0, 0.0])
    x02 = np.array([0.0, 0.0])

    sys1 = LPV(A1, C1, B1, D1)
    sys2 = LPV(A2, C2, B2, D2)

    print("Équivalents ?", sys1.isEquivalentTo(sys2, x01, x02))
    
    print("\n---------- TEST 3 : Reachability reduction on dLPV ----------")

    nx = 2  
    nu = 1 
    ny = 1 
    np_ = 2
    
    A = np.zeros((nx, nx, np_))
    B = np.zeros((nx, nu, np_))
    C = np.ones((ny, nx))  

    A[:, :, 0] = np.array([[1.0, 0.0],
                           [0.0, 1.0]])
    A[:, :, 1] = np.array([[0.5, 0.0],
                           [0.0, 0.5]])
    
    B[:, :, 0] = np.array([[1.0],
                           [0.0]])
    B[:, :, 1] = np.array([[0.0],
                           [1.0]])

    D_mat = np.zeros((ny, nu))  

    x0 = np.array([0.1, 0.2])

    system = dLPV(A, C, B, D_mat)

    Reach_mat, red_order, Ar, Br, Cr, x0r = system.reach_reduction(x0)

    print("Reachability matrix Reach_mat:\n", Reach_mat)
    print("Reduced order:", red_order)
    for i in range(np_):
        print(f"Ao[:,:,{i}]:\n", Ar[:, :, i])
    print("Reduced Bo (3D):")
    for i in range(np_):
        print(f"Bo[:,:,{i}]:\n", Br[:, :, i])
    print("Reduced Cr:\n", Cr)
    print("Reduced initial state x0r:\n", x0r)
    
    A2 = np.array([[0.4472, 0.8944, 0.4472, 0.8944, 0.2236, 0.4472],
                [0.8944, -0.4472, 0.8944, -0.4472, 0.4472, -0.2236]])
    
    O = orth(A2)
    print("O =", O)
    U, s, Vh = np.linalg.svd(A2, full_matrices=False)
    print("U =", U)
    print("s", s)
    print("Vh", Vh)
    
    print("\n---------- TEST 4 : Observability Reduction on dLPV ----------")

    nx = 2    
    nu = 1    
    ny = 1    
    np_ = 2   

    A = np.zeros((nx, nx, np_))
    B = np.zeros((nx, nu, np_))
    C = np.array([[1.0, 0.0]])  
    D = np.zeros((ny, nu))    

    A[:, :, 0] = np.array([[1.0, 0.1],
                           [0.0, 1.0]])
    A[:, :, 1] = np.array([[0.9, 0.2],
                           [0.0, 0.95]])

    B[:, :, 0] = np.array([[1.0],
                           [0.0]])
    B[:, :, 1] = np.array([[0.0],
                           [1.0]])
    x0 = np.array([0.1, 0.2])
    system = dLPV(A, C, B, D)

    Obs_mat, red_order, Ar, Br, Cr, x0r = system.obs_reduction(x0)

    print("Observability matrix Obs_mat:\n", Obs_mat)
    print("Reduced order (observability):", red_order)
    print("Reduced Ao (3D):")
    for i in range(np_):
        print(f"Ao[:,:,{i}]:\n", Ar[:, :, i])
    print("Reduced Bo (3D):")
    for i in range(np_):
        print(f"Bo[:,:,{i}]:\n", Br[:, :, i])
    print("Reduced Co:\n", Cr)
    print("Reduced initial state x0o:\n", x0r)
    
    print("\n---------- TEST 5 : Minimal Realization of dLPV ----------")
    
    minimal_sys, x0_min = system.minimize(x0)
    print("Minimal realization:")
    print("Original order:", nx)
    print("Minimal order:", minimal_sys.nx)
    print("Reduced initial state:", x0_min.flatten())

    for i in range(minimal_sys.np):
        print(f"Amin[:,:,{i}]:\n", minimal_sys.A[:, :, i])
    for i in range(minimal_sys.np):
        print(f"Bmin[:,:,{i}]:\n", minimal_sys.B[:, :, i])
    print("Cmin:\n", minimal_sys.C)
    
    
    print("---------- TEST 6 : isIsomorphic ----------")

    nx = 2
    nu = 1
    ny = 1
    np_ = 2

    A1 = np.zeros((nx, nx, np_))
    B1 = np.zeros((nx, nu, np_))
    C1 = np.array([[1.0, 0.0]])
    D1 = np.zeros((ny, nu))

    A1[:, :, 0] = np.array([[1.0, 0.1],
                            [0.0, 0.9]])
    A1[:, :, 1] = np.array([[0.8, 0.2],
                            [0.1, 0.7]])
    B1[:, :, 0] = np.array([[1.0],
                            [0.0]])
    B1[:, :, 1] = np.array([[0.0],
                            [1.0]])

    T = np.array([[2.0, 1.0],
                  [0.0, 1.0]])
    T_inv = np.linalg.inv(T)

    A2 = np.zeros_like(A1)
    B2 = np.zeros_like(B1)
    for i in range(np_):
        A2[:, :, i] = T @ A1[:, :, i] @ T_inv
        B2[:, :, i] = T @ B1[:, :, i]
    C2 = C1 @ T_inv
    D2 = D1.copy()
    
    for i in range(np_):
        print(f"A1[:,:,{i}]:\n", A1[:, :, i])
    for i in range(np_):
        print(f"B1[:,:,{i}]:\n", B1[:, :, i])
    print("C1 = ", C2)
    print("D1 = ", D2)
    
    for i in range(np_):
        print(f"A2[:,:,{i}]:\n", A2[:, :, i])
    for i in range(np_):
        print(f"B2[:,:,{i}]:\n", B2[:, :, i])
    print("C2 = ", C2)
    print("D2 = ", D2)
    

    x0 = np.array([0.0, 0.0])
    x0_other = np.array([0.0, 0.0])

    sys1 = dLPV(A1, C1, B1, D1)
    sys2 = dLPV(A2, C2, B2, D2)

    # Test isomorphisc
    result_iso = sys1.isIsomorphic(sys2, x0, x0_other)
    print("Isomorphic (expected True):", result_iso)

    # test not isomorphic
    A3 = np.zeros((3, 3, np_))
    B3 = np.zeros((3, nu, np_))
    C3 = np.array([[1.0, 0.0, 0.0]])
    D3 = np.zeros((ny, nu))

    sys3 = dLPV(A3, C3, B3, D3)
    x0_other3 = np.zeros(3)

    result_non_iso = sys1.isIsomorphic(sys3, x0, x0_other3)
    print("Isomorphic (expected False):", result_non_iso)
    
    print("---------- TEST 7 : Recursion function ----------")

    nx = 2
    ny = 1
    np_ = 2

    # Define system matrices A, G, C for dLPV instance
    A = np.zeros((nx, nx, np_))
    G = np.zeros((nx, ny, np_))
    C = np.array([[1.0, 0.0]])

    # Fill A and G for two modes
    A[:, :, 0] = np.array([[0.9, 0.1],
                          [0.0, 0.8]])
    A[:, :, 1] = np.array([[0.7, 0.2],
                          [0.1, 0.6]])
    G[:, :, 0] = np.array([[0.1],
                          [0.0]])
    G[:, :, 1] = np.array([[0.05],
                          [0.02]])

    # Noise covariance matrices for measurement noise T_sig (shape ny x ny x np)
    T_sig = np.zeros((ny, ny, np_))
    T_sig[:, :, 0] = np.array([[10.0]])
    T_sig[:, :, 1] = np.array([[5.0]])

    # Probability weights for each mode (np x 1)
    psig = np.array([[0.6], [0.4]])

    # Create instance of your dLPV class
    sys = dLPV(A, C, None, None)
    sys.G = G
    sys.nx = nx
    sys.ny = ny
    sys.np = np_

    # Run recursion
    Pold, Qold, Kold = sys.Recursion(T_sig, psig)

    print("Pold matrices:")
    for i in range(np_):
        print(f"Mode {i}:\n", Pold[:, :, i])

    print("Qold matrices:")
    for i in range(np_):
        print(f"Mode {i}:\n", Qold[:, :, i])

    print("Kold matrices:")
    for i in range(np_):
        print(f"Mode {i}:\n", Kold[:, :, i])


if __name__ == "__main__":
    main()
