import numpy as np

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

N_OUT = 1000
N_IN = 1000
N_EXPERIMENTS = 1000

def f(X):
    Y = np.ones(X.shape[0])
    Y[np.sum(X**2,1)-0.6<0] = -1
    # flip 10% of Y
    Y[np.random.permutation(len(Y))[0:int(len(Y) / 10)]] *= -1
    return Y


W = np.zeros((N_EXPERIMENTS,6))
E_in = np.zeros(N_EXPERIMENTS)
E_out = np.zeros(N_EXPERIMENTS)

def X2Z(X):
    Z = np.column_stack((X, X[:, 0] * X[:, 1], X**2))
    Z_b = np.column_stack((np.ones(Z.shape[0]), Z))
    return (Z, Z_b)

for i in range(N_EXPERIMENTS):
    X_in = np.random.uniform(-1.0, 1.0, (N_OUT, 2))
    (Z_in, Z_in_b) = X2Z(X_in)
    Y_in_f = f(X_in)

    W[i] = np.matmul(np.matmul(np.linalg.inv(np.matmul(Z_in_b.T, Z_in_b)),Z_in_b.T),Y_in_f)
    def g(X):
        Y = np.ones(X.shape[0])
        (Z, Z_b) = X2Z(X)
        Y[np.matmul(Z_b, W[i])<0]=-1
        return Y

    Y_in_g = g(X_in)
    E_in[i] = 1.0-np.sum(Y_in_g==Y_in_f)/N_IN

    print(W[i])

    X_out = np.random.uniform(-1.0, 1.0, (N_OUT, 2))
    (Z_out, Z_out_b) = X2Z(X_out)
    Y_out_f = f(X_out)
    Y_out_g = g(X_out)
    E_out[i] = 1.0 - np.sum(Y_out_g == Y_out_f) / N_OUT

print("mean of W:", np.mean(W,0))
print("mean of E_in:", np.mean(E_in))
print("mean of E_out:", np.mean(E_out))
