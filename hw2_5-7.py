import numpy as np

N_OUT = 1000
N_IN = 10
N_EXPERIMENTS = 1000

# create N_OUT points
p_out = np.random.uniform(-1.0, 1.0, (N_OUT, 2))

# choose random two points and define f
(p1, p2) = p_out[np.random.permutation(N_OUT)[0:2]]
# x_2 = m*x_1+b
m = (p1[1] - p2[1]) / (p1[0] - p2[0])
b = p1[1] - m * p1[0]
def f(X):
    Y = np.ones(X.shape[0])
    Y[X[:,1]<m*X[:,0]+b] = -1
    return Y

X = p_out
Y = f(X)

E_in = np.zeros(N_EXPERIMENTS)
iters = np.zeros(N_EXPERIMENTS)

for i in range(N_EXPERIMENTS):

    X_in = X[np.random.permutation(N_OUT)[0:N_IN]]
    X_in_b = np.c_[np.ones(X_in.shape[0]), X_in]
    Y_in_f = f(X_in)
    W = np.matmul(np.matmul(np.linalg.inv(np.matmul(X_in_b.T, X_in_b)),X_in_b.T),Y_in_f)
#    print(-W[1]/W[2], -W[0]/W[2])

    def g(X):
        Y = np.ones(X.shape[0])
        X_b = np.c_[np.ones(X.shape[0]), X]
        Y[np.matmul(X_b, W)<0]=-1
        return Y

    Y_in_g = g(X_in)
    E_in[i] = 1.0-np.sum(Y_in_g==Y_in_f)/N_IN

    ########
    # PLA
    ########
    #W = np.array([0.0,0.0,0.0])
    def p(X):
        Y = np.ones(X.shape[0])
        X_b = np.c_[np.ones(X.shape[0]), X]
        Y[np.matmul(X_b, W)<0]=-1
        return Y

    j = 0
    while True:

        j += 1
        #time.sleep(0.1)

        Y_in_h = p(X_in) # output
        mpi = np.where(Y_in_f != Y_in_h)[0]
        if (len(mpi)>0):
            # perceptron rule
            k = np.random.randint(len(mpi))
            W += Y_in_f[mpi[k]]*X_in_b[mpi[k]]
        else:
            iters[i] = j
            break

print(np.mean(E_in))
print(np.mean(iters))

E_out = np.zeros(N_EXPERIMENTS)
for i in range(N_EXPERIMENTS):
    X_out = np.random.uniform(-1.0, 1.0, (N_OUT, 2))
    Y_out_f = f(X_out)
    Y_out_g = g(X_out)
    E_out[i] = 1.0 - np.sum(Y_out_f == Y_out_g) / N_OUT

print(np.mean(E_out))
