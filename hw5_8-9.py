import numpy as np
from numpy import exp, log

#
def logistic(z):
    return 1/(1+exp(-z))
a = logistic
#
def d_logistic(z):
    return logistic(z)*(1-logistic(z))
d_a = d_logistic


#
def cross_entropy(y_, y):
    return np.mean(-(y_*log(y)+(1-y_)*log(1-y)))
E = cross_entropy
#
def d_cross_entropy(y_, x, w):
    y = g(x, w)
    return (y - y_)*x
d_E = d_cross_entropy


def g(w, X):
    return a(np.dot(X, w))

RUN = 100
N = 100
Nout = 1000
LR = 0.01
Eout = np.empty(RUN)
Epochs = np.empty(RUN)

for run in range(RUN):

    # create f = mx + b
    [f_p1_x, f_p1_y], [f_p2_x, f_p2_y] = np.random.random((2, 2)) * 2 - 1
    m = (f_p1_y - f_p2_y) / (f_p1_x - f_p2_x)
    b = f_p1_y - m * f_p1_x
    print("m, b of f: m:%f, b:%f" % (m, b))

    def f(X):
        # x[0] is bias
        X = np.atleast_2d(X)
        return np.asarray((m * X[:, 1] + b <= X[:, 2]), dtype=np.int)

    # create X, Xout
    X = np.column_stack((np.ones(N), np.random.random((N, 2))))
    Xout = np.column_stack((np.ones(Nout), np.random.random((Nout, 2))))

    #
    w = np.zeros(3)
    epoch = 0

    while True:
        epoch += 1
        ns = np.random.permutation(N)
        w_prev_epoch = w.copy()

        for i in range(N):
            x = X[ns[i]]
            y_ = f(x)
            w -= LR * d_E(y_, x, w)

        if np.sqrt(np.sum((w - w_prev_epoch)**2)) < 0.01:
            Epochs[run] = epoch
            Eout[run] = E(f(Xout), g(w, Xout))
            print('run:', run, 'epoch:', epoch, "w:", w, "Ein:", E(f(X), g(w, X)), "Eout:", E(f(Xout), g(w, Xout)))
            break

# problem 8, 9
print(Eout.mean())
print(Epochs.mean())
