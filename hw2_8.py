import numpy as np

N_OUT = 10000
N_IN = 1000
N_EXPERIMENTS = 1000

def f(X):
    Y = np.ones(X.shape[0])
    Y[np.sum(X**2,1)-0.6<0] = -1
    return Y

X = np.random.uniform(-1.0, 1.0, (N_OUT, 2))
Y = f(X)

# flip 10% of Y
Y[np.random.permutation(N_OUT)[0:int(N_OUT/10)]] *= -1

E_in = np.zeros(N_EXPERIMENTS)

for i in range(N_EXPERIMENTS):

    X_in = X[np.random.permutation(N_OUT)[0:N_IN]]
    X_in_b = np.c_[np.ones(X_in.shape[0]), X_in]
    Y_in_f = f(X_in)
    W = np.matmul(np.matmul(np.linalg.inv(np.matmul(X_in_b.T, X_in_b)),X_in_b.T),Y_in_f)

    def g(X):
        Y = np.ones(X.shape[0])
        X_b = np.c_[np.ones(X.shape[0]), X]
        Y[np.matmul(X_b, W)<0]=-1
        return Y

    Y_in_g = g(X_in)
    E_in[i] = 1.0-np.sum(Y_in_g==Y_in_f)/N_IN

print(np.mean(E_in))
