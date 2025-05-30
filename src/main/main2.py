import numpy as np
from src.main.config import initialize_parameters
from src.lpv.LPV import LPV
from src.lpv.asLPV import asLPV

params = initialize_parameters()

np.random.seed(1)
Ntot = params['Ntot']
ny, np_, nx, nu = params['ny'], params['np'], params['nx'], params['nu']
A, K, C, F, D = params['A'], params['K'], params['C'], params['F'], params['D']
u,v,psig,p = params['u'],params['v'],params['psig'],params['p']
N, Nval, Ntot = params['N'], params['Nval'], params['Ntot']


# print("Test : Simulate_Innovation // VERIFIED ")
# B = np.zeros((2, 1, 2)) 
# #B[:, :, 0] = np.array([[0], [0]])
# #B[:, :, 1] = np.array([[0], [0]])
# 
# D = np.zeros((1,1))
# u2 = np.zeros((1,Ntot))
# 
# print(u.shape)
# 
# LpvSys = LPV(A,C,B=B,D=D,K=K,F=F)
# (y,ynf,x) = LpvSys.simulate_y(u2,v,p,Ntot)
# print("y :" ,y)
# # ynew = np.array([
# #     [0.9794, -0.2656, -0.1535, -0.0159, -1.4287,
# #      -0.7147, 1.3786, -3.5134, 0.8502, 0.0]
# # ])
# # 
# # pnew = np.array([
# #     [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
# #     [0.0387, -0.1185, -0.4488, -1.2149, -0.1990, 0.6277, -1.1521, -1.2657, -0.3922, -1.3991]
# # ])
# 
# err = LpvSys.simulate_Innovation(Ntot,y,p)
# print("innov")
# print(err)
# 
# print("Test 1 : IsFormInnovation is True")
# asLPVsys = asLPV(A,C,K,F)
# 
# print(asLPVsys.isFormInnovation(psig))
# 
# print("Test  2 : IsFormInnovation is False beacuse of F not identity")
# asLPVsys = asLPVsys = asLPV(A,C,K,np.array([[2]]))
# print(asLPVsys.isFormInnovation(psig))
# 
# print("Test 3 : IsFormInnovation is False because of Unstable Matrixs eigval >= 1")
# 
# A1= np.zeros((2, 2, 2))
# A1[:, :, 0] = np.array([[1.2, 0.0],
#                        [0.0, 1.1]])
# A1[:, :, 1] = np.array([[0.9, 0.4],
#                        [0.3, 0.8]])
# 
# K1 = np.zeros((2, 1, 2))
# K1[:, :, 0] = np.array([[2.0],
#                        [-1.0]])
# K1[:, :, 1] = np.array([[1.5],
#                        [1.5]])
# 
# C1 = np.array([[1.0, 1.0]])
# F1 = np.array([[1.0]])
# psig1 = np.array([0.5, 0.5])
# 
# 
# asLPVsys1 = asLPV(A1,C1,K1,F1)
# print(asLPVsys1.isFormInnovation(psig1))
# 
# print("Test 4 : Compute_vesp")
# asLPVsys = asLPV(A,C,K,F)
# 
# v_test = np.array([[0.9794, -0.2656, -0.5484, -0.0963, -1.3807, -0.7284,1.8860 ,-2.9414,0.9800 ,-1.1918]])
# expected = 1.83663002
# 
# print(asLPVsys.compute_vsp(v_test))
print("Test 5 : Computing Qi")
asLPVsys = asLPV(A,C,K,F)

v_test = np.array([[0.9794, -0.2656, -0.5484, -0.0963, -1.3807, -0.7284,1.8860 ,-2.9414,0.9800 ,-1.1918]])
        
        #Randomness is used to defined p, so we define one for this test
p_test = np.array([
    [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
    [0.0387, -0.1185, -0.4488, -1.2149, -0.1990, 0.6277, -1.1521, -1.2657, -0.3922, -1.3991]
])

Q = asLPVsys.compute_Qi(v_test,p_test)
print(Q[0][0][0])
print(Q[0][0][0] == 1.836630022)

print("Test 6 : Computing Pi")
asLPVsys = asLPV(A,C,K,F)

v_test = np.array([[0.9794, -0.2656, -0.5484, -0.0963, -1.3807, -0.7284,1.8860 ,-2.9414,0.9800 ,-1.1918]])
        
        #Randomness is used to defined p, so we define one for this test
p_test = np.array([
    [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
    [0.0387, -0.1185, -0.4488, -1.2149, -0.1990, 0.6277, -1.1521, -1.2657, -0.3922, -1.3991]
])

print(Q)
P = asLPVsys.compute_Pi(psig,Q)
print(P)

