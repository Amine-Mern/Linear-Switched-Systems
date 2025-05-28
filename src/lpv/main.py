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

if __name__ == "__main__":
    main()
