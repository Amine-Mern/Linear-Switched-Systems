import numpy as np
from LPV import LPV

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

    sys1 = LPV(A1, B1, C1, D1)
    sys2 = LPV(A2, B2, C2, D2)

    print("Équivalents ?", sys1.isEquivalentTo(sys2, x01, x02), "\n")

    print("---------- TEST 2 : Non equivalent systems ----------")
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

    sys1 = LPV(A1, B1, C1, D1)
    sys2 = LPV(A2, B2, C2, D2)

    print("Équivalents ?", sys1.isEquivalentTo(sys2, x01, x02))

if __name__ == "__main__":
    main()
