import numpy as np
import matplotlib.pyplot as plt
from quadprog import solve_qp

X = np.array([[1, 0], [0, 1], [0, -1], [-1, 0], [0, 2], [0, -2], [-2, 0]],
             dtype=np.double)
Y = np.array([[-1], [-1], [-1], [1], [1], [1], [1]],
             dtype=np.double)

assert len(X) == len(Y)

N = len(Y)


def MK(X_):
    return (1 + X_.dot(X_.T))**2

G = np.outer(Y, Y) * MK(X)
a = np.ones(N, dtype=np.double)
C = np.array(np.c_[Y, np.eye(N, N)], dtype=np.double)
b = np.zeros(1 + N, dtype=np.double)

# to suppress positive definite error
G += np.eye(N, N) * 1e-13

alphas, _, _, _, _, _ = solve_qp(G, a, C, b, meq=1)

print(alphas)
print(np.sum(alphas > 1e-3))
