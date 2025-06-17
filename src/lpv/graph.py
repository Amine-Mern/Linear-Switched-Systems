import numpy as np
import plotext as plt

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

def main():
    A = np.zeros((2,2,2))
    A[:, :, 0] = np.array([[0.4, 0.4], [0.2, 0.1]])
    A[:, :, 1] = np.array([[0.1, 0.1], [0.2, 0.3]])
        
    C = np.array([[1, 0]])
        
    K = np.zeros((2, 1, 2))
    K[:, :, 0] = np.array([[0], [1]])
    K[:, :, 1] = np.array([[0], [1]])
        
    F = np.array([[1]])
        
    asLPV_sys = asLPV(A,C,K,F)
    
    Ntot = 1000
    ny = asLPV_sys.ny
    
    v = calculate_v(ny,Ntot)
    p = calculate_p(Ntot)
    psig = calculate_psig(p)
    
    y,ynf,x = asLPV_sys.simulate_y(v,p)
    
    as_min_system, Qmin = asLPV_sys.stochMinimize(v,p,psig)
    
    #print("IsInFormInnovation : ",asLPV_sys.isFormInnovation(psig))
    
    error = asLPV_sys.simulate_Innovation(y,p)
    
    half = Ntot//2
    
    innov_error = error[:,np.floor(half)+1:]
    
# Pls don't delete these comments, might be useful (this is a unfinished version of the graph) (amine)
#     print(p[:,np.floor(half)+1:])
#     
#     print("----------------------")
#     p[:,half+1:-1]

    y2,ynf2,x2 = asLPV_sys.simulate_y(innov_error,p[:,np.floor(half)+1:])
    x_half = list(range(0,half))
    x_full = list(range(Ntot))
    
    plt.plot(x_half,y[0,half:],label='Output of S')
    
    plt.plot(x_half,y2[0,:],label='Output of S^m')
        
    err = y[0,np.floor(half)+1:] - y2[0, :]
    plt.plot(x_half, err, label='Error', color ='black')
    
    plt.title('Outputs of S and S^m using mu')
    plt.xlabel('Sample index k')
    plt.ylabel('Value')
    
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
