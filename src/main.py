import numpy as np
from config import initialize_parameters
from lpv.LPV import LPV

params = initialize_parameters()

np.random.seed(1)
Ntot = params['Ntot']
ny, np_, nx, nu = params['ny'], params['np'], params['nx'], params['nu']
A, K, C, F, D = params['A'], params['K'], params['C'], params['F'], params['D']
u,v,psig,p = params['u'],params['v'],params['psig'],params['p']
N, Nval, Ntot = params['N'], params['Nval'], params['Ntot']



print("Test : Simulate_Innovation // VERIFIED ")
B = np.zeros((2, 1, 2)) 
#B[:, :, 0] = np.array([[0], [0]])
#B[:, :, 1] = np.array([[0], [0]])

D = np.zeros((1,1))
u2 = np.zeros((1,Ntot))

print(u.shape)

LpvSys = LPV(A,C,B=B,D=D,K=K,F=F)
(y,ynf,x) = LpvSys.simulate_y(u2,v,p,Ntot)
print("y :" ,y)
# ynew = np.array([
#     [0.9794, -0.2656, -0.1535, -0.0159, -1.4287,
#      -0.7147, 1.3786, -3.5134, 0.8502, 0.0]
# ])
# 
# pnew = np.array([
#     [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
#     [0.0387, -0.1185, -0.4488, -1.2149, -0.1990, 0.6277, -1.1521, -1.2657, -0.3922, -1.3991]
# ])

err = LpvSys.simulate_Innovation(Ntot,y,p)
print("innov")

print(err)