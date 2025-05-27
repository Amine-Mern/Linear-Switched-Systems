import numpy as np

def initialize_parameters() :
    """
    Initializes and returns system parameters for an LPV model
    
    Returns:
        dict: A dictionary containing the following keys:
            - A (np.ndarray): State transition matrices. 
            - K (np.ndarray): Disturbance gain matrices. 
            - C (np.ndarray): Output matrix.
            - F (np.ndarray): Noise matrix. 
            - D (np.ndarray): Input-output gain matrix.
            - nx (int): Number of states.
            - ny (int): Number of outputs.
            - nu (int): Number of inputs.
            - np (int): Number of scheduling parameters.
            - N (int): Number of training samples.
            - Nval (int): Number of validation samples.
            - Ntot (int): Total number of samples (N + Nval).
    """
    
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
    np_ = K.shape[2]
    nx = A.shape[0]
    ny = D.shape[0]
    nu = D.shape[1]
    u = calculate_u(nu,Ntot)
    v = calculate_v(ny,Ntot)
    p = calculate_p(Ntot)
    psig = calculate_psig(np_,Ntot,p)
    return {
        'A': A,
        'K': K,
        'C': C,
        'F': F,
        'D': D,
        'nx': nx,
        'ny': ny,
        'nu': ny,
        'np': np_,
        'u':u,
        'v':v,
        'psig':psig,
        'p':p,
        'N': N,
        'Nval': Nval,
        'Ntot': Ntot
    }

def calculate_u(nu,Ntot):
    return 3*np.random.rand(nu,Ntot) - 1.5

def calculate_v(ny,Ntot):
    return np.random.randn(ny,Ntot)

def calculate_p(Ntot):
    p = np.zeros((2,Ntot))
    
    p[0,:] = np.ones(Ntot)
    
    p[1,:] =  3 * np.random.rand(Ntot) - 1.5
    
    return p

def calculate_psig(np_,Ntot,p):
    psig = np.zeros((np_, 1))
    for i in range(np_):
        psig[i, 0] = np.var(p[i, :]) + np.mean(p[i, :])**2
        
    return psig