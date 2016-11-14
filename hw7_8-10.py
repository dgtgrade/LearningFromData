# PLA code is copied from hw2_5-7 and modified

import numpy as np
from quadprog import solve_qp
import matplotlib.pyplot as plt

N_OUT = 3000
N_IN = 100
N_EXPERIMENTS = 1000

experiment = 0
win_svm = 0
win_pla = 0
total_sv = 0

while experiment < N_EXPERIMENTS:

    # choose random two points and define f
    [p1, p2] = np.random.uniform(-1.0, 1.0, (2, 2))
    # x_2 = m*x_1+b
    m = (p1[1] - p2[1]) / (p1[0] - p2[0])
    b = p1[1] - m * p1[0]

    def f(X):
        Y = np.ones(X.shape[0])
        Y[X[:, 1] < m * X[:, 0] + b] = -1
        return Y

    X_in = np.random.uniform(-1.0, 1.0, (N_IN, 2))
    X_in_b = np.c_[np.ones(X_in.shape[0]), X_in]
    Y_in_f = f(X_in)

    def Wtomb(W):
        return -W[1] / W[2], -W[0] / W[2]

    #
    if not (-N_IN < np.sum(Y_in_f) < N_IN):
        continue

    experiment += 1

    # PLA

    W_pla = np.array([0.0,0.0,0.0])

    def p(X):
        Y = np.ones(X.shape[0])
        X_b = np.c_[np.ones(X.shape[0]), X]
        Y[np.matmul(X_b, W_pla)<0]=-1
        return Y

    while True:

        Y_in_h = p(X_in) # output
        mpi = np.where(Y_in_f != Y_in_h)[0]
        if (len(mpi)>0):
            # perceptron rule
            k = np.random.randint(len(mpi))
            W_pla += Y_in_f[mpi[k]]*X_in_b[mpi[k]]
        else:
            break

    # SVM
    _G = np.array(np.outer(Y_in_f, Y_in_f) * X_in.dot(X_in.T), dtype=np.double)
    _a = np.ones(X_in.shape[0], dtype=np.double)
    _C = np.array(np.c_[Y_in_f, np.eye(Y_in_f.shape[0], Y_in_f.shape[0])], dtype=np.double)
    _b = np.zeros(1 + Y_in_f.shape[0], dtype=np.double)

    # to suppress positive definite error
    _G += np.eye(X_in.shape[0], X_in.shape[0]) * 1e-13

    alpha, _, _, _, _, _ = solve_qp(_G, _a, _C, _b, meq=1)

    W_svm = np.sum(alpha[alpha>0] * Y_in_f[alpha>0] * X_in[alpha>0].T, axis=1)
    # print(alpha[alpha>0])

    sv0 = np.argmax(alpha)
    b_svm = 1/Y_in_f[sv0] - W_svm.T.dot(X_in[sv0])
    W_svm = np.hstack((b_svm, W_svm))
    num_sv = np.sum(alpha > 0.1)
    total_sv += num_sv

    def g(X, W):
        Y = np.ones(X.shape[0])
        X_b = np.c_[np.ones(X.shape[0]), X]
        Y[np.matmul(X_b, W) < 0] = -1
        return Y

    X_out = np.random.uniform(-1.0, 1.0, (N_OUT, 2))
    Y_out_f = f(X_out)

    Y_out_pla = g(X_out, W_pla)
    Y_out_svm = g(X_out, W_svm)
    acc_pla = np.sum(Y_out_f == Y_out_pla) / N_OUT
    acc_svm = np.sum(Y_out_f == Y_out_svm) / N_OUT

    # Plot
    if False:
        pla_m, pla_b = Wtomb(W_pla)
        svm_m, svm_b = Wtomb(W_svm)

        def draw_line(l_m, l_b, color):
            lx = [-1, 1]
            ly = [-l_m+l_b, l_m+l_b]
            plt.plot(lx, ly, marker="", linewidth=3, color=color)

        plt.scatter(X_in[:, 0], X_in[:, 1], s=100, c=Y_in_f, cmap="bwr")

        draw_line(m, b, "black")
        draw_line(pla_m, pla_b, "blue")
        draw_line(svm_m, svm_b, "red")

        axes = plt.gca()
        axes.set_xlim([-1, 1])
        axes.set_ylim([-1, 1])
        plt.show()

    print("acc_pla:", acc_pla, "acc_svm:", acc_svm, "num_sv:", num_sv)
    if acc_svm > acc_pla:
        win_svm += 1
    else:
        win_pla += 1

print("win_pla", win_pla, "win_svm:", win_svm, "num_sv:", total_sv/N_EXPERIMENTS)
