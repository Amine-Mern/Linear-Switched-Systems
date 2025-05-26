import numpy as np
from config import initialize_parameters
from lpv.LPV import LPV

params = initialize_parameters()

np.random.seed(1)
Ntot = params['Ntot']
ny, np_, nx, nu = params['ny'], params['np'], params['nx'], params['nu']
A, K, C, F, D = params['A'], params['K'], params['C'], params['F'], params['D']
N, Nval, Ntot = params['N'], params['Nval'], params['Ntot']

print("Test : Simulate_Innovation")
LpvSys = LPV(A,np.zeros,C,K=K,F=F)
LpvSys.simulate_Innovation(Ntot,y,p)