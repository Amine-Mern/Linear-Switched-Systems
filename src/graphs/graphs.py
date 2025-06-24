import numpy as np
import plotext as plt
import math

from src.lpv.asLPV import asLPV

def calculate_v(ny, Ntot):
    return np.random.randn(ny,Ntot)
    
def calculate_psig(p):
    np_, Ntot = p.shape
    psig = np.zeros((np_, 1))
    for i in range(np_):
        psig[i, 0] = np.var(p[i, :]) + np.mean(p[i, :])**2
    return psig

def calculate_p(Ntot):
    p = np.zeros((2,Ntot))
    
    p[0,:] = np.ones(Ntot)
    
    p[1,:] =  3 * np.random.rand(Ntot) - 1.5
    
    return p

def calculate_p_2(Ntot):
    p_2 = np.zeros((2,Ntot))
    
    p_2[0,:] = np.ones(Ntot)
    p_2[1,:] = math.sqrt(12) * np.random.rand(Ntot) - (math.sqrt(12)/2)
    
    return p_2

def main(asLPV_sys):
    
    Ntot = 1200
    ny = asLPV_sys.ny
    
    v = calculate_v(ny,Ntot)
    p = calculate_p(Ntot)
    psig = calculate_psig(p)
    
    # Graph 1
    y,ynf,x = asLPV_sys.simulate_y(v,p)
    
    as_min_system,Qmin = asLPV_sys.stochMinimize(v,p,psig)

    error = as_min_system.simulate_Innovation(y,p)
    
    half = Ntot//2
    
    innov_error = error[:,half:]

    y2,ynf2,x2 = as_min_system.simulate_y(innov_error,p[:,-half:])
    x_half = list(range(0,half))
    x_full = list(range(Ntot))
      
    plt.plot(y[0,half:],label='Output of S')
    
    plt.plot(y2[0,:],label='Output of S^m')
        
    err = y[0,half:] - y2[0, :]
    
    plt.plot(err, label='Error', color ='black')
    
    plt.title('Outputs of S and S^m using mu')
    plt.xlabel('Sample index k')
    plt.ylabel('Value')
    
    plt.grid(True)
    plt.show()
    
    #Clear  Graph
    plt.clear_figure()
    
    # Graph 2
    print("Graph 2")
    p_2 = calculate_p_2(Ntot)
    
    y,ynf,x = asLPV_sys.simulate_y(v,p_2)
    y2,ynf2,x2 = asLPV_sys.simulate_y(innov_error,p_2[:,-half:])
    
    plt.plot(x_half,y[0,half:],label='Output of S')
    
    plt.plot(x_half,y2[0,:],label='Output of S^m')
        
    err = y[0,half:] - y2[0, :]
    plt.plot(x_half, err, label='Error', color ='black')
    
    plt.title('Outputs of S and S^m using mu\'')
    plt.xlabel('Sample index k')
    plt.ylabel('Value')
    
    plt.grid(True)
    plt.show()
    
    # Clearing for the next graph
    plt.clear_figure()


def build_asLPV_example_1():
    A = np.zeros((2,2,2))
    A[:, :, 0] = np.array([[0.4, 0.4], [0.2, 0.1]])
    A[:, :, 1] = np.array([[0.1, 0.1], [0.2, 0.3]])
        
    C = np.array([[1, 0]])
        
    K = np.zeros((2, 1, 2))
    K[:, :, 0] = np.array([[0], [1]])
    K[:, :, 1] = np.array([[0], [1]])
        
    F = np.array([[1]])
    return asLPV(A,C,K,F)


def build_asLPV_example_2():

    A = np.zeros((3, 3, 2))
    A[:, :, 0] = np.array([
        [0.4, 0.4, 0.0],
        [0.2, 0.1, 0.0],
        [0.0, 0.0, 0.2]
    ])
    A[:, :, 1] = np.array([
        [0.1, 0.1, 0.0],
        [0.2, 0.3, 0.0],
        [0.0, 0.0, 0.2]
    ])

    C = np.array([[10, 0, 0]])

    K = np.zeros((3, 1, 2))
    K[:, :, 0] = np.array([[0], [1], [1]])
    K[:, :, 1] = np.array([[0], [1], [1]])

    F = np.array([[1]])

    return asLPV(A,C,K,F)

def build_asLPV_example_3():
    A = np.zeros((2, 2, 2))
    A[:, :, 0] = np.array([[0.4, 0.4], [0.2, 0.1]])
    A[:, :, 1] = np.array([[0.1, 0.1], [0.2, 0.3]])

    C = np.array([[10, 0]])

    K = np.zeros((2, 1, 2))
    K[:, :, 0] = np.array([[0], [1]])
    K[:, :, 1] = np.array([[0], [1]])

    F = np.array([[1]])
    return asLPV(A,C,K,F)    
    
if __name__ == "__main__":
    
    #First System
    print("First System Stably Invertable")
    main(build_asLPV_example_1())

    print("Second System Unstably Invertable, not clear if is in Innovation Form")
    print("S^m is not minimal")
    #Second System
    main(build_asLPV_example_2())
    
    print("Third System Unstable Invertable, not clear if is in Innovation Form")
    print("S^m is minimal")
    #Third System
    main(build_asLPV_example_3())