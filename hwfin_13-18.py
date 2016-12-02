import numpy as np
import matplotlib.pyplot as plt
from quadprog import solve_qp


# function f
def f(X):
    Y = np.sign(X[:, 1] - X[:, 0] + 0.25 * np.sin(np.pi * X[:, 0]))
    Y[Y == 0] = 1
    return Y


# Kernel function
def Kernel(X1: np.ndarray, X2: np.ndarray, gamma):
    return np.exp(-gamma * np.sum((X1 - X2) ** 2, axis=max(X1.ndim, X2.ndim) - 1))


# show training points
def show_training_points(X, Y):
    plt.clf()
    plt.scatter(X[Y == 1, 0], X[Y == 1, 1], edgecolor='blue',
                marker='o', facecolors='none', s=100, linewidth=1)
    plt.scatter(X[Y == -1, 0], X[Y == -1, 1], facecolor='red',
                marker='x', s=100, linewidth=1)
    plt.show()
    plt.waitforbuttonpress()


# show clusters
def show_clusters(X, clusters, centroids):
    colors = np.array([
        (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
        (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207)
    ])/255
    assert len(colors) >= len(centroids)
    plt.clf()
    plt.scatter(X[:, 0], X[:, 1], edgecolor=colors[clusters],
                marker='o', facecolor='none', s=50, linewidth=1)
    plt.scatter(centroids[:, 0], centroids[:, 1], facecolor=colors,
                marker='+', s=200, linewidth=2)
    plt.show()
    plt.waitforbuttonpress()


def lloyd(X, k):
    n = X.shape[0]
    d = X.shape[1]

    # 0 initialize centroids
    centroids = np.random.rand(k, d) * 2 - 1

    while True:
        # 1 set a cluster of each X to a nearest centroid
        XX = np.tile(X.reshape(n, 1, 2), (1, k, 1))
        CC = np.tile(centroids.reshape(1, k, d), (n, 1, 1))
        clusters = np.argmin(np.sqrt(np.sum((XX - CC) ** 2, axis=2)), axis=1)
        if np.sum(np.bincount(clusters, minlength=k) == 0) > 0:
            raise ValueError("zero-size cluster occurred")

        # show_clusters(X, clusters, centroids)
        # 2 move each centroid to center of each cluster
        centroids_prev = centroids.copy()
        centroids = np.array(
            [np.mean(X[clusters == k, :], axis=0) for k in range(k)]
        )
        # show_clusters(X, clusters, centroids)

        # 3 check if converged
        if np.sum((centroids - centroids_prev) ** 2) == 0:
            # plt.close()
            break

    return centroids


#
# Regular Form
#
def RunRegular(X: np.ndarray, Y: np.ndarray, X_test: np.ndarray, Y_test: np.ndarray, k, gamma):

    n = len(X)
    n_test = len(X_test)

    try:
        centroids = lloyd(X, k)
    except ValueError:
        raise ValueError("Failed on lloyd")
    
    X1 = np.tile(X.reshape(n, 1, 2), (1, k, 1))
    C1 = np.tile(centroids.reshape(1, k, 2), (n, 1, 1))
    Phi = np.c_[np.ones(n), Kernel(X1, C1, gamma)]
    W = np.linalg.inv(Phi.T.dot(Phi)).dot(Phi.T).dot(Y)
    
    def g_reg(xs):
        xs1 = np.tile(xs.reshape(len(xs), 1, 2), (1, k, 1))
        cs1 = np.tile(centroids.reshape(1, k, 2), (len(xs), 1, 1))
        zs = np.array(np.c_[
                          np.ones(len(xs1)),
                          Kernel(xs1, cs1, gamma)
                      ])
        r = np.sign(zs.dot(W))
        r[r == 0] = 1
        return r
    
    Y_reg = g_reg(X)
    Y_test_reg = g_reg(X_test)
    E_in = 1 - np.sum(Y == Y_reg) / n
    E_out = 1 - np.sum(Y_test == Y_test_reg) / n_test
    
    return [E_in, E_out]


#
# Kernel Form
#
def RunKernel(X: np.ndarray, Y: np.ndarray, X_test: np.ndarray, Y_test: np.ndarray, gamma):
    
    n = len(X)
    n_test = len(X_test)
    
    # hard-margin SVM
    X1 = np.tile(X.reshape(n, 1, 2), (1, n, 1))
    X2 = X1.transpose((1, 0, 2))
    G = np.outer(Y, Y) * Kernel(X1, X2, gamma)
    a = np.ones(n, dtype=np.double)
    C = np.array(np.c_[Y, np.eye(n, n)], dtype=np.double)
    b = np.zeros(1 + n, dtype=np.double)
    # to suppress positive definite error
    salt = 1e-13
    sugar = 1e-3
    G += np.eye(n, n) * salt
    alphas, _, _, _, _, _ = solve_qp(G, a, C, b, meq=1)
    # print(alphas)
    # print(np.sum(alphas > sugar))

    # hypothesis g
    def g_svm(xs):
        sv_is = alphas > sugar
        sv_ix = np.argmax(alphas)
        b_ = Y[sv_ix] - np.sum(alphas[sv_is] * Y[sv_is] * Kernel(X[sv_is], X[sv_ix], gamma))
        r = np.empty(len(xs))
        for i in range(len(xs)):
            t = np.sum(alphas[sv_is] * Y[sv_is] * Kernel(X[sv_is], xs[i], gamma))
            r[i] = np.sign(t + b_)
        r[r == 0] = 1
        return r
    Y_svm = g_svm(X)
    Y_test_svm = g_svm(X_test)
    E_in = 1 - np.sum(Y == Y_svm) / n
    E_out = 1 - np.sum(Y_test == Y_test_svm) / n_test
    if E_in > 0:
        raise ValueError("Failed on hard-margin SVM")
    return [E_in, E_out]


#
#
#
def main():

    N_TRAIN = 100  # number of training points
    N_TEST = 1000
    RUN = 1000  # number of experiments

    plt.ion()

    # problem 13-17
    print("Problem 13-17")
    print("Running Experiments...")
    RegularK9_E_out = np.empty(RUN)
    RegularK9_E_in = np.empty(RUN)
    RegularK12_E_out = np.empty(RUN)
    RegularK12_E_in = np.empty(RUN)
    RegularG2_E_out = np.empty(RUN)
    RegularG2_E_in = np.empty(RUN)
    Kernel_E_in = np.empty(RUN)
    Kernel_E_out = np.empty(RUN)
    fail_svm = 0
    run = 0
    while True:

        X_train = np.random.rand(N_TRAIN, 2) * 2 - 1
        Y_train = f(X_train)

        X_test = np.random.rand(N_TEST, 2) * 2 - 1
        Y_test = f(X_test)

        # show_training_points()

        try:
            RegularK9_E_in[run], RegularK9_E_out[run] =\
                RunRegular(X_train, Y_train, X_test, Y_test, k=9, gamma=1.5)
            RegularK12_E_in[run], RegularK12_E_out[run] = \
                RunRegular(X_train, Y_train, X_test, Y_test, k=12, gamma=1.5)
            RegularG2_E_in[run], RegularG2_E_out[run] = \
                RunRegular(X_train, Y_train, X_test, Y_test, k=9, gamma=2)
        except ValueError as e:
            print(e)
            continue

        try:
            Kernel_E_in[run], Kernel_E_out[run] = RunKernel(X_train, Y_train, X_test, Y_test, gamma=1.5)
        except Exception as e:
            print(e)
            fail_svm += 1
            continue

        print("Run #%d," % run,
              "RegularK9 E_in: %3.3f E_out: %3.3f," % (RegularK9_E_in[run], RegularK9_E_out[run]),
              "RegularK12 E_in: %3.3f E_out: %3.3f," % (RegularK12_E_in[run], RegularK12_E_out[run]),
              "RegularG2 E_in: %3.3f E_out: %3.3f," % (RegularG2_E_in[run], RegularG2_E_out[run]),
              "Kernel E_in: %3.3f E_out: %3.3f" % (Kernel_E_in[run], Kernel_E_out[run]))
        run += 1
        if run == RUN:
            break

    print("Complete")

    print("P13. Not separable by the RBF Kernel: %4.3f%% of the time" %
          (fail_svm/(RUN + fail_svm) * 100))

    print("P14. Kernel (K=9) form beats Regular form in terms of E_out: %4.3f%% of the time" %
          ((Kernel_E_out < RegularK9_E_out).sum() / RUN * 100))

    print("P15. Kernel (K=12) form beats Regular form in terms of E_out: %4.3f%% of the time" %
          ((Kernel_E_out < RegularK12_E_out).sum() / RUN * 100))

    print("P16. go from Kernel (K=9) to Kernel (K=12)",
          "E_in down and E_out down: %4.3f%%" %
          ((np.all(
              [RegularK9_E_in > RegularK12_E_in, RegularK9_E_out > RegularK12_E_out],
              axis=0).sum()) / RUN * 100))

    print("P17. go from Kernel (g=1.5) to Kernel (g=2.0)")
    cmp_in = np.sign(RegularG2_E_in - RegularK9_E_in)
    cmp_out = np.sign(RegularG2_E_out - RegularK9_E_out)
    p17_down_down = np.all([cmp_in == -1, cmp_out == -1], axis=0).sum() / RUN * 100
    p17_down_up = np.all([cmp_in == -1, cmp_out == 1], axis=0).sum() / RUN * 100
    p17_up_down = np.all([cmp_in == 1, cmp_out == -1], axis=0).sum() / RUN * 100
    p17_up_up = np.all([cmp_in == 1, cmp_out == 1], axis=0).sum() / RUN * 100
    print("\tE_in down and E_out down: %4.3f%%" % p17_down_down)
    print("\tE_in down and E_out up: %4.3f%%" % p17_down_up)
    print("\tE_in up and E_out down: %4.3f%%" % p17_up_down)
    print("\tE_in up and E_out up: %4.3f%%" % p17_up_up)

    print("P18. Regular form zero E_in: %4.3f%% of the time" %
          ((RegularK9_E_in == 0).sum() / RUN * 100))


if __name__ == '__main__':
    main()
