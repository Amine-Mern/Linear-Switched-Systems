import numpy as np

def initialize_parameters() :
    A = np.zeros((2,2,2))
    A[:, :, 0] = np.array([[0.4, 0.4], [0.2, 0.1]])
    A[:, :, 1] = np.array([[0.1, 0.1], [0.2, 0.3]])
    K = np.zeros((2, 1, 2))
    K[:, :, 0] = np.array([[0], [1]])
    K[:, :, 1] = np.array([[0], [1]])
    C = np.array([[1, 0]])
    F = np.array([[1]])
    D = np.array([[1]])
    N = 1000
    Nval = 1000
    Ntot = N + Nval
    
    return {
        'A': A,
        'K': K,
        'C': C,
        'F': F,
        'D': D,
        'nx': A.shape[0],
        'ny': C.shape[0],
        'nu': D.shape[1],
        'np': A.shape[2],
        'N': N,
        'Nval': Nval,
        'Ntot': Ntot
    }